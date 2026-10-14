"""Check measured rheology, physical scaling, and hydraulic feedback paths."""

from dataclasses import asdict
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from analyze_i060_mechanism import resistance_budget, sample_profiles
from i060_coupled import (Fields, Problem, grad, hydraulic_coefficients,
                          material_problem, objective, quadrature, resistance_q)
from i060_material import pchip


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/"config/i060_material_delivery.json"


class MaterialTests(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        torch.set_num_threads(2)
        torch.manual_seed(7)
        self.problem = material_problem(CONFIG, inlet="dilution")
        self.material = self.problem.material

    def test_recipe_and_shared_dimensional_scales(self):
        material = self.material
        self.assertAlmostEqual(self.problem.cin, 1/7.2)
        self.assertAlmostEqual(self.problem.cin*self.problem.dilution_fraction, 1/16.5)
        self.assertEqual(material.reference_viscosity_pa_s, .0142)
        self.assertAlmostEqual(material.reference_flux_m_s, .0002347417840375587)
        self.assertAlmostEqual(material.reference_time_s, 1278.0)
        restored = Problem(**json.loads(json.dumps(asdict(self.problem))))
        self.assertEqual(asdict(restored), asdict(self.problem))

    def test_budget_preserves_water_limit_and_deposit_penalty(self):
        x = np.linspace(0, 1, 11)
        times = np.array([0., .2, .8])
        c = np.zeros((len(x), len(times)))
        sigma = np.broadcast_to([0., .04, .04], c.shape)
        budget = resistance_budget(x, c, sigma, times, self.problem)
        water = self.material.water_viscosity_pa_s/self.material.reference_viscosity_pa_s
        np.testing.assert_allclose(budget["slurry_excess"], 0., atol=1e-15)
        np.testing.assert_allclose(budget["total"], water*(1-sigma[0]/self.problem.phi0)**-3)
        self.assertAlmostEqual(budget["reconstructed_flow"][0], self.material.initial_flow)
        self.assertLess(budget["reconstructed_flow"][1], budget["reconstructed_flow"][0])
        self.assertAlmostEqual(budget["reconstructed_flow"][1], budget["reconstructed_flow"][2])

    def test_budget_matches_local_darcy_coefficients_at_interpolated_events(self):
        x = np.array([0., .1, .4, 1.])
        source_times = np.array([0., .2, .7])
        fields = {"times": source_times,
                  "c": .1*(1-x[:, None])*source_times,
                  "sigma": .04*(1-.5*x[:, None])*source_times}
        times = np.array([.1, .35])
        state = sample_profiles(fields, times)
        np.testing.assert_allclose(state["c"], .1*(1-x[:, None])*times)
        budget = resistance_budget(x, **state, times=times, problem=self.problem)
        c, sigma = (torch.tensor(state[key]) for key in ("c", "sigma"))
        density, gravity = hydraulic_coefficients(self.problem, c, sigma, torch.tensor(times)[None, :])
        np.testing.assert_allclose(budget["total"], np.trapezoid(density.numpy(), x, axis=0))
        np.testing.assert_allclose(budget["gravity_head"], np.trapezoid(gravity.numpy(), x, axis=0))

    def test_interpolation_reproduces_all_reported_viscosities(self):
        source = json.loads((ROOT/"data/yang_2020/rheology.json").read_text(encoding="utf-8"))
        c = torch.tensor([self.material.solid_fraction(ratio)
                          for ratio in source["water_cement_mass_ratios"]])
        for age, values in zip(source["hydration_ages_min"], source["dynamic_viscosity_pa_s"]):
            t = torch.full_like(c, age*60/self.material.reference_time_s)
            torch.testing.assert_close(self.material.viscosity(c, t), torch.tensor(values))

    def test_water_limit_and_interpolation_support(self):
        c = torch.linspace(0, self.problem.concentration_ceiling, 137)[:, None]
        times = torch.linspace(0, 1, 83)[None, :]
        c, times = torch.broadcast_tensors(c, times)
        c, times = c.clone().requires_grad_(True), times.clone().requires_grad_(True)
        mu = self.material.viscosity(c, times)
        dc, dt = torch.autograd.grad(mu.sum(), (c, times))
        self.assertTrue(torch.isfinite(dc).all() and torch.isfinite(dt).all())
        self.assertGreaterEqual(float(mu.min().detach()), self.material.water_viscosity_pa_s-1e-15)
        # Within this case's measured age range, both directions are monotone.
        self.assertGreaterEqual(float(dc.min()), -1e-12)
        self.assertGreaterEqual(float(dt.min()), -1e-12)
        torch.testing.assert_close(mu[0], torch.full_like(mu[0], .00089))
        torch.testing.assert_close(dt[0], torch.zeros_like(dt[0]), atol=1e-15, rtol=0)

    def test_interpolant_has_matching_first_derivatives_at_knots(self):
        knots = torch.tensor([0., .1, .3, .6, 1.])
        values = torch.tensor([1., 1.2, 2., 2., 2.3])
        for point in knots[1:-1]:
            x = torch.tensor([float(point)-1e-8, float(point)+1e-8], requires_grad=True)
            derivative = grad(pchip(knots, values, x), x)
            torch.testing.assert_close(derivative[0], derivative[1], atol=1e-6, rtol=1e-6)

    def test_initial_pressure_drive_uses_water_gravity_and_viscosity(self):
        net = Fields(self.problem)
        times = torch.zeros(2, 1, requires_grad=True)
        flow = resistance_q(net, times, quadrature(32))
        expected = (1-1000*9.81*.3/100000)*.0142/.00089
        torch.testing.assert_close(flow, torch.full_like(flow, expected))
        torch.testing.assert_close(net.q(times), flow)
        torch.testing.assert_close(grad(flow, times), torch.zeros_like(times), atol=1e-12, rtol=0)
        x = torch.linspace(0, 1, 11)[:, None]
        xt = torch.cat((x, 0*x), 1).requires_grad_(True)
        c, sigma, _ = net(xt)
        density, gravity = hydraulic_coefficients(self.problem, c, sigma, xt[:, 1:])
        darcy = expected+(grad(net.p(xt), xt)[:, :1]+gravity)/density
        torch.testing.assert_close(darcy, torch.zeros_like(darcy), atol=1e-12, rtol=0)

    def test_flow_uses_the_local_suspension_left_in_the_column(self):
        class PrescribedStates:
            def __init__(self, problem, retained_high_feed):
                self.problem = problem
                self.retained_high_feed = retained_high_feed

            def transport_components(self, xt):
                x = xt[:, :1]
                low = self.problem.cin*self.problem.dilution_fraction
                c = low+0*x
                if self.retained_high_feed:
                    c = c+(self.problem.cin-low)*4*x*(1-x)
                return c, 0*x

            def deposition_grid(self, x, times):
                sigma = torch.full((len(x), len(times)), .04)
                return sigma, torch.zeros_like(sigma)

        times = torch.tensor([[.8]])
        low_flow = resistance_q(PrescribedStates(self.problem, False), times, quadrature(64))
        mixed_flow = resistance_q(PrescribedStates(self.problem, True), times, quadrature(64))
        self.assertLess(float(mixed_flow), float(low_flow))

    def test_hydraulic_gradient_reaches_concentration_and_deposition(self):
        net = Fields(self.problem)
        with torch.no_grad():
            net.hazard[-1].bias[1] = 2.
        times = torch.tensor([[.13], [.45], [.85]])
        for name, bias, index in (("concentration", net.state[-1].bias, 2),
                                  ("capture", net.hazard[-1].bias, 1)):
            derivative = torch.autograd.grad(resistance_q(net, times, quadrature(32)).sum(), bias)[0]
            with self.subTest(output=name), torch.no_grad():
                old, step = float(bias[index]), 1e-5
                bias[index] = old+step
                plus = resistance_q(net, times, quadrature(32)).sum()
                bias[index] = old-step
                minus = resistance_q(net, times, quadrature(32)).sum()
                bias[index] = old
                self.assertLess(float(derivative[index]), 0)
                torch.testing.assert_close(derivative[index], (plus-minus)/(2*step), rtol=1e-6, atol=1e-8)

    def test_learned_hydraulics_allow_growth_and_quadratic_startup(self):
        net = Fields(self.problem)
        with torch.no_grad():
            for parameter in net.flow.parameters():
                parameter.zero_()
            net.flow[-1].bias.fill_(.3)
        t0 = torch.zeros(1, 1, requires_grad=True)
        q0 = net.q(t0)
        qt0 = grad(q0, t0)
        qtt0 = grad(qt0, t0)
        self.assertGreater(float(q0.detach()), 1)
        self.assertGreater(float(net.q(torch.tensor([[.4]])).detach()), float(q0.detach()))
        torch.testing.assert_close(qt0, torch.zeros_like(qt0))
        self.assertGreater(float(qtt0.detach()), 0)

    def test_both_losses_backpropagate_with_material_coupling(self):
        for method in ("resistance", "augmented"):
            with self.subTest(method=method):
                net = Fields(self.problem)
                loss = objective(net, torch.tensor([.05, .4, .9]),
                                 torch.tensor([[0.], [.02], [.55], [.8]]), method, quadrature(16))
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                for parameter in net.optimization_parameters(method):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
