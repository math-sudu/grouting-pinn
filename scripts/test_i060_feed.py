"""Check the coupled flux lift and the physical supply control volume."""

from dataclasses import replace
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from analyze_i060_feed import compare_cement_accounts, material_account, prepared_material
from analyze_i060_segment_tasks import evaluate_segment_task
from i060_coupled import (Fields, Problem, continued_problem, evaluate, grad, material_problem, objective, quadrature,
                          resistance_q, residuals, segment_inventory_residual)


CONFIG = Path(__file__).resolve().parents[1]/"config/i060_material_delivery.json"


class FeedTests(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        torch.set_num_threads(2)
        torch.manual_seed(7)
        self.problem = material_problem(CONFIG, inlet="water")
        self.net = Fields(self.problem)

    def test_constant_recipes_share_the_stock_clock_and_preserve_their_prefix(self):
        times = torch.tensor([[0.], [.002], [.1], [.55], [1.]])
        for weights in ((0., 1., 0.), (.5, .5, 0.), (.25, .5, .25)):
            problem = material_problem(CONFIG, initial_recipe=weights)
            self.assertEqual(problem.material, self.problem.material)
            self.assertEqual(problem.transitions, [])
            torch.testing.assert_close(problem.recipe(times), times.new_tensor([weights]).expand(5, -1))
            feed = problem.feed_weights(times)
            concentration = problem.cin*(weights[0]+problem.dilution_fraction*weights[1])
            torch.testing.assert_close(problem.inlet_c(times), problem.startup(times)*concentration)
            torch.testing.assert_close(feed[:, :2], problem.startup(times)*times.new_tensor([weights[:2]]))
            torch.testing.assert_close(feed.sum(1), torch.ones(5))
            branch = continued_problem(problem, .25, "water")
            before = torch.linspace(0, .25, 11)[:, None]
            torch.testing.assert_close(branch.feed_weights(before), problem.feed_weights(before))
            torch.testing.assert_close(branch.feed_weights(torch.tensor([[.5]])), times.new_tensor([[0., 0., 1.]]))
        for weights in ((0., 0., 0.), (1., -.1, .1), (1., 0.), (float("nan"), 0., 0.)):
            with self.assertRaises(ValueError):
                Problem(initial_recipe=weights)

    def test_off_grid_water_start_has_zero_subsequent_cement_input(self):
        problem = replace(self.problem, switch_time=.2375)
        net = Fields(problem)
        _, fields = evaluate(net, "resistance", quadrature(8),
                             spatial_points=11, temporal_points=11)
        end = problem.transitions[-1]["start"]+problem.switch_width
        self.assertIn(end, fields["times"])
        np.testing.assert_array_equal(fields["feed_c"][fields["times"] >= end], 0.)
        supply = json.loads(CONFIG.read_text(encoding="utf-8"))
        before = prepared_material(fields, problem, supply, end)
        after = prepared_material(fields, problem, supply, end+.2)
        self.assertAlmostEqual(before["fed_cement_kg"], after["fed_cement_kg"], places=14)
        net_flux = fields["j"][0]-fields["j"][-1]
        accumulated = np.r_[0., np.cumsum(np.diff(fields["times"])*(net_flux[1:]+net_flux[:-1])/2)]
        np.testing.assert_allclose(fields["mass_defect"],
                                   fields["mobile"]+fields["deposited"]-accumulated)

    def test_boundary_flux_and_outlet_for_both_hydraulic_maps(self):
        t = torch.tensor([[0.], [.003], [.2], [.55], [.58], [.9]])
        for method in ("resistance", "augmented"):
            q = resistance_q(self.net, t, quadrature(32)) if method == "resistance" else self.net.q(t)
            xt = torch.cat((0*t, t), 1).requires_grad_(True)
            c, _, jd = self.net.transport(xt, q)
            torch.testing.assert_close(q*c+jd, q*self.problem.inlet_c(t), rtol=1e-14, atol=1e-16)
            self.assertGreater(float(c[-1].detach()), 0.)
            self.assertEqual(float(self.problem.inlet_c(t)[-1]), 0.)
            xt = torch.cat((0*t+1, t), 1).requires_grad_(True)
            c, _, jd = self.net.transport(xt, q)
            torch.testing.assert_close(grad(c, xt)[:, :1], 0*t, rtol=0, atol=1e-15)
            torch.testing.assert_close(jd, 0*t, rtol=0, atol=0)

    def test_capture_can_stop_exactly_without_losing_irreversibility(self):
        with torch.no_grad():
            for parameter in self.net.hazard.parameters():
                parameter.zero_()
            self.net.hazard[-1].bias[:4] = -1.
        t = torch.tensor([[0.], [.02], [4/64], [.2], [1.]])
        xt = torch.cat((torch.full_like(t, .4), t), 1).requires_grad_(True)
        sigma = self.net(xt)[1]
        rate = grad(sigma, xt)[:, 1:]
        self.assertGreaterEqual(float(sigma.detach().diff(dim=0).min()), 0.)
        torch.testing.assert_close(rate[2:], torch.zeros_like(rate[2:]), rtol=0, atol=0)
        grid_sigma, grid_rate = self.net.deposition_grid(torch.tensor([.4]), t)
        torch.testing.assert_close(grid_sigma.flatten(), sigma.flatten())
        torch.testing.assert_close(grid_rate.flatten(), rate.flatten())
        self.assertGreater(float(sigma[2].detach()), 0.)

    def test_feed_control_is_a_nonnegative_recipe_mixture(self):
        times = torch.linspace(0, 1, 501)[:, None]
        concentrations = torch.tensor([self.problem.cin,
                                      self.problem.cin*self.problem.dilution_fraction, 0.])
        for history in ("steady", "dilution", "water"):
            p = replace(self.problem, inlet=history)
            weights = p.feed_weights(times)
            self.assertGreaterEqual(float(weights.min()), -1e-15)
            torch.testing.assert_close(weights.sum(1), torch.ones(len(times)))
            torch.testing.assert_close(weights@concentrations, p.inlet_c(times).flatten())

    def test_initial_states_and_capture_rate(self):
        x = torch.linspace(0, 1, 13)[:, None]
        xt = torch.cat((x, 0*x), 1).requires_grad_(True)
        c, sigma, jd = self.net.transport(xt, torch.full_like(x, self.problem.material.initial_flow))
        for field in (c, sigma, jd, grad(sigma, xt)[:, 1:]):
            torch.testing.assert_close(field, 0*x, rtol=0, atol=1e-15)

    def test_lift_preserves_total_storage_identity_and_backward_paths(self):
        x = torch.tensor([0., .15, .5, 1.])
        t = torch.tensor([[.02], [.55], [.9]])
        xx, tt = torch.meshgrid(x, t.flatten(), indexing="ij")
        xt = torch.stack((xx.flatten(), tt.flatten()), 1).requires_grad_(True)
        q = resistance_q(self.net, t, quadrature(32)).T.expand(len(x), -1).reshape(-1, 1)
        c, sigma, jd = self.net.transport(xt, q)
        p = self.problem
        total = (p.phi0-sigma)*c+sigma
        expected = grad(total, xt)[:, 1:]+q*grad(c, xt)[:, :1]+grad(jd, xt)[:, :1]
        r = residuals(self.net, x, t, "resistance", quadrature(32))
        torch.testing.assert_close(expected, p.cin*r["mass"]+p.retention*p.cin*r["exchange"])
        for method in ("resistance", "augmented"):
            self.net.zero_grad(set_to_none=True)
            objective(self.net, x, t, method, quadrature(32)).backward()
            for parameter in self.net.optimization_parameters(method):
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_unused_stock_is_charged_and_water_balance_includes_initial_water(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config["feed"]["unused_stock_reserve_m3"] = [.0001, .0002]
        t = np.array([0., .5, 1.])
        # An exactly conservative uniform synthetic state with no outlet solids.
        weights = np.tile([.4, .3, .3], (3, 1))
        cf = .4*self.problem.cin+.3*self.problem.cin*self.problem.dilution_fraction
        stored = cf*t
        fields = {"x": np.array([0., 1.]), "times": t,
                  "q": np.ones(3), "c": np.zeros((2, 3)),
                  "sigma": np.tile(stored, (2, 1)),
                  "j": np.array([[cf]*3, [0.]*3]), "feed_weights": weights}
        task = {"endpoint_phase": "deposited", "segment_edges": [0., 1.], "target_segments": [1],
                "minimum_mean_solid_fraction": [cf*.75], "deadline": 1.}
        stop = evaluate_segment_task(fields, self.problem.phi0, task)["first_stop"]
        account = material_account(fields, self.problem, config, stop, task["segment_edges"])
        self.assertAlmostEqual(stop["time"], .75)
        np.testing.assert_allclose(account["unused_stock_litre"], [.1, .2])
        self.assertAlmostEqual(account["fresh_cement_kg"], account["fed_cement_kg"]+account["unused_cement_kg"])
        self.assertAlmostEqual(account["fresh_water_kg"], account["stock_water_kg"]+account["separate_water_kg"])
        self.assertAlmostEqual(account["cement_balance_defect_kg"], 0.)
        self.assertAlmostEqual(account["segment_capture_defect_kg"][0], -account["segment_cement_kg"][0])
        self.assertAlmostEqual(account["feed_to_face_defect_kg"], 0.)
        self.assertAlmostEqual(account["initial_column_water_kg"]+account["fresh_water_kg"],
                               account["column_water_at_stop_kg"]+account["outlet_water_kg"]+account["unused_water_kg"])

    def test_saving_partition_keeps_signed_destinations_and_balance_defect_separate(self):
        reference = {"time": .8, "supply": {
            "segment_deposited_cement_kg": [.1, .2, .3],
            "segment_mobile_cement_kg": [.01, .02, .03],
            "outlet_cement_kg": .04, "unused_cement_kg": .02,
            "fresh_cement_kg": .725, "cement_balance_defect_kg": .005}}
        selected = {"time": .6, "supply": {
            "segment_deposited_cement_kg": [.12, .12, .22],
            "segment_mobile_cement_kg": [.005, .005, .01],
            "outlet_cement_kg": .08, "unused_cement_kg": .05,
            "fresh_cement_kg": .606, "cement_balance_defect_kg": -.004}}
        comparison = compare_cement_accounts(reference, selected, [1, 3])
        self.assertEqual(comparison["reference_time"], .8)
        self.assertEqual(comparison["selected_time"], .6)
        expected = {"target_deposited": .06, "outside_deposited": .08,
                    "target_mobile": .025, "outside_mobile": .015,
                    "outlet": -.04, "unused_preparation": -.03}
        for name, value in expected.items():
            self.assertAlmostEqual(comparison["destinations"][name]["reduction_kg"], value)
        self.assertAlmostEqual(comparison["balance_defect_reduction_kg"], .009)
        self.assertAlmostEqual(comparison["prepared_cement_reduction_kg"], .119)
        self.assertAlmostEqual(comparison["prepared_cement_reduction_percent"], 100*.119/.725)
        self.assertAlmostEqual(
            sum(row["reduction_kg"] for row in comparison["destinations"].values())
            +comparison["balance_defect_reduction_kg"],
            comparison["prepared_cement_reduction_kg"])

    def test_segment_constraint_integrates_flux_and_reaches_state_parameters(self):
        problem = self.problem

        class UniformCapture:
            def __init__(self):
                self.problem = problem
                self.rate = torch.tensor(.01, requires_grad=True)

            def transport_components(self, xt, flow=None):
                t = xt[:, 1:]
                return 0*t, .01*(1-xt[:, :1]) if flow is not None else 0*t

            def deposition_grid(self, x, times):
                sigma = self.rate*times.T.expand(len(x), -1)
                return sigma, torch.ones_like(sigma)*self.rate

            def q(self, t):
                return torch.ones_like(t)

        net = UniformCapture()
        times = torch.tensor([[0.], [.1], [.35], [.7], [1.]])
        edges = torch.linspace(0, 1, 7)
        r = segment_inventory_residual(net, times, edges, .01, "augmented", quadrature(16))
        torch.testing.assert_close(r, torch.zeros_like(r), rtol=0, atol=1e-15)
        derivative = torch.autograd.grad(r.sum(), net.rate)[0]
        self.assertAlmostEqual(float(derivative), 6*float(times.sum())/.01)
        phases = segment_inventory_residual(net, times, edges, .01, "augmented", quadrature(16), include_capture=True)
        torch.testing.assert_close(phases[0], r)
        torch.testing.assert_close(phases[1], times.T.expand(6, -1))
        r = segment_inventory_residual(self.net, times, edges, .01, "resistance", quadrature(16), include_capture=True)
        r.square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all()
                            for p in self.net.optimization_parameters("resistance")))


if __name__ == "__main__":
    unittest.main()
