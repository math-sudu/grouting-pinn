"""Verify state support, conserved storage and hydraulic feedback gradients."""

import copy
import math
import unittest

import torch

from i060_coupled import Fields, Problem, grad, objective, quadrature, resistance_q, residuals


class CoupledTests(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        torch.set_num_threads(2)
        torch.manual_seed(7)
        self.problem = Problem(inlet="dilution")
        self.net = Fields(self.problem)

    def test_initial_and_boundary_conditions(self):
        x = torch.linspace(0, 1, 13)[:, None]
        t = torch.linspace(0, 1, 17)[:, None]
        c, s, f = self.net(torch.cat((x, 0*x), 1))
        self.assertTrue(torch.equal(c, 0*c) and torch.equal(s, 0*s))
        inlet, _, _ = self.net(torch.cat((0*t, t), 1))
        torch.testing.assert_close(inlet, self.problem.inlet_c(t))
        xt = torch.cat((0*t+1, t), 1).requires_grad_(True)
        c, _, f = self.net(xt)
        self.assertEqual(float(grad(c, xt)[:, 0].detach().abs().max()), 0)
        self.assertEqual(float(f.detach().abs().max()), 0)

    def test_dilution_can_leave_higher_concentration_inside(self):
        with torch.no_grad():
            for parameter in self.net.state.parameters():
                parameter.zero_()
            # At late times the inlet is .02; the interior can still hold .08.
            self.net.state[-1].bias[2] = math.log(.08/.92)-math.log(.02/.98)
        c, _, _ = self.net(torch.tensor([[1., .9], [0., .9]]))
        self.assertAlmostEqual(float(c[0].detach()), .08, places=9)
        self.assertAlmostEqual(float(c[1].detach()), .02, places=9)

    def test_capture_clock_preserves_zero_initial_rates(self):
        t = torch.linspace(0, 1, 17)[:, None].requires_grad_(True)
        torch.testing.assert_close(grad(self.problem.capture_clock(t), t),
                                   self.problem.startup(t))
        x = torch.linspace(0, 1, 13)[:, None]
        xt = torch.cat((x, 0*x), 1).requires_grad_(True)
        _, sigma, _ = self.net(xt)
        for field in (sigma, self.net.p(xt)):
            torch.testing.assert_close(grad(field, xt)[:, 1:], 0*x,
                                       rtol=0, atol=1e-15)
        t0 = torch.zeros(1, 1, requires_grad=True)
        for flow in (self.net.q(t0), resistance_q(self.net, t0, quadrature(32))):
            torch.testing.assert_close(grad(flow, t0), 0*t0, rtol=0, atol=1e-15)
        # The leading deposition term scales cubically with startup time.
        times = torch.tensor([[1e-5], [2e-5]])
        _, sigma, _ = self.net(torch.cat((0*times+.5, times), 1))
        self.assertAlmostEqual(float((sigma[1]/sigma[0]).detach()), 8., places=3)

    def test_outlet_curvature_is_not_suppressed_by_boundary_maps(self):
        class PolynomialRawFields(torch.nn.Module):
            def forward(self, coordinates):
                x = (coordinates[:, :1]+1)/2
                return torch.cat((1+2*x, x, .3+0*x), 1)

        self.net.state = PolynomialRawFields()
        xt = torch.tensor([[1., .7]], requires_grad=True)
        c, _, diffusive_flux = self.net(xt)
        cx = grad(c, xt)[:, :1]
        cxx = grad(cx, xt)[:, :1]
        cxxx = grad(cxx, xt)[:, :1]
        flux_xx = grad(grad(diffusive_flux, xt)[:, :1], xt)[:, :1]
        self.assertEqual(float(cx.detach()), 0)
        self.assertEqual(float(diffusive_flux.detach()), 0)
        self.assertNotEqual(float(cxx.detach()), 0)
        self.assertNotEqual(float(cxxx.detach()), 0)
        self.assertNotEqual(float(flux_xx.detach()), 0)

    def test_total_storage_identity(self):
        x = torch.tensor([.15, .48, .85])
        t = torch.tensor([[.17], [.56], [.87]])
        r = residuals(self.net, x, t, "resistance", quadrature(32))
        xx, tt = torch.meshgrid(x, t.flatten(), indexing="ij")
        xt = torch.stack((xx.flatten(), tt.flatten()), 1).requires_grad_(True)
        c, s, f = self.net(xt)
        q = resistance_q(self.net, t, quadrature(32)).T.expand(len(x), -1).reshape(-1, 1)
        total = (self.problem.phi0-s)*c+s
        total_residual = grad(total, xt)[:, 1:]+q*grad(c, xt)[:, :1]+grad(f, xt)[:, :1]
        summed = self.problem.cin*r["mass"]+self.problem.retention*self.problem.cin*r["exchange"]
        torch.testing.assert_close(summed, total_residual)

    def test_resistance_gradient_reaches_deposition_parameters(self):
        t = torch.tensor([[.2], [.5], [.9]])
        parameter = self.net.hazard[-1].bias
        derivative = torch.autograd.grad(resistance_q(self.net, t, quadrature(48)).sum(), parameter)[0][1]
        with torch.no_grad():
            old = float(parameter[1])
            step = 1e-5
            parameter[1] = old+step
            plus = resistance_q(self.net, t, quadrature(48)).sum()
            parameter[1] = old-step
            minus = resistance_q(self.net, t, quadrature(48)).sum()
            parameter[1] = old
        torch.testing.assert_close(derivative, (plus-minus)/(2*step), rtol=1e-6, atol=1e-8)

    def test_optimizer_contains_exactly_the_parameters_used_by_each_loss(self):
        x = torch.tensor([.15, .48, .85])
        t = torch.tensor([[.17], [.56], [.87]])
        for method in ("resistance", "differential", "augmented"):
            with self.subTest(method=method):
                self.net.zero_grad(set_to_none=True)
                objective(self.net, x, t, method, quadrature(16)).backward()
                parameters = tuple(self.net.optimization_parameters(method))
                used = {id(p) for p in self.net.parameters() if p.grad is not None}
                self.assertEqual({id(p) for p in parameters}, used)

    def test_capture_is_irreversible_and_uses_the_integrated_rate(self):
        # Strongly varying positive nodal rates exercise partial intervals and
        # exact knots, including the zero-startup and terminal boundaries.
        with torch.no_grad():
            for parameter in self.net.hazard.parameters():
                parameter.zero_()
            self.net.hazard[-1].bias.copy_(4*torch.sin(torch.arange(65)*1.7))
        times = torch.unique(torch.cat((torch.linspace(0, 1, 513),
                                        torch.linspace(0, 1, 65))))[:, None]
        xt = torch.cat((torch.full_like(times, .37), times), 1).requires_grad_(True)
        _, sigma, _ = self.net(xt)
        self.assertGreaterEqual(float(sigma.detach().diff(dim=0).min()), 0.)
        self.assertEqual(float(sigma[0].detach()), 0.)
        self.assertLess(float(sigma.detach().max()), self.problem.capacity)
        rates = torch.nn.functional.softplus(self.net.hazard[-1].bias.detach())
        clock = self.problem.capture_clock(times)
        index = (torch.searchsorted(self.net.capture_grid, clock.flatten(), right=True)-1).clamp(0, 63)
        fraction = (clock.flatten()-self.net.capture_grid[index])/self.net.capture_grid.diff()[index]
        expected = self.problem.startup(times)*(rates[index]+fraction*(rates[index+1]-rates[index]))[:, None]
        torch.testing.assert_close(grad(self.net.cumulative_hazard(xt), xt)[:, 1:], expected)
        torch.testing.assert_close(grad(sigma, xt)[:, 1:],
                                   (self.problem.capacity-sigma)*expected)
        middle = torch.tensor([[.37, .2387]], requires_grad=True)
        epsilon = torch.tensor([[0., 1e-6]])
        finite = (self.net(middle+epsilon)[1]-self.net(middle-epsilon)[1])/(2e-6)
        torch.testing.assert_close(grad(self.net(middle)[1], middle)[:, 1:], finite,
                                   rtol=1e-7, atol=1e-9)

    def test_tensor_grid_matches_pointwise_capture_and_autograd(self):
        x = torch.tensor([0., .37, 1.])
        times = torch.tensor([[0.], [1/64], [.2387], [1.]])
        xx, tt = torch.meshgrid(x, times.flatten(), indexing="ij")
        xt = torch.stack((xx.flatten(), tt.flatten()), 1).requires_grad_(True)
        _, sigma, _ = self.net(xt)
        grid_sigma, grid_rate = self.net.deposition_grid(x, times)
        torch.testing.assert_close(grid_sigma.flatten(), sigma.flatten())
        torch.testing.assert_close(grid_rate.flatten(), grad(sigma, xt)[:, 1])
        parameter = self.net.hazard[-1].weight
        torch.testing.assert_close(torch.autograd.grad(grid_sigma.sum(), parameter)[0],
                                   torch.autograd.grad(sigma.sum(), parameter)[0])

    def test_state_only_lbfgs_preserves_the_coupled_state_update(self):
        reduced = copy.deepcopy(self.net)
        x = torch.tensor([.15, .48, .85])
        t = torch.tensor([[.17], [.56], [.87]])
        for net, parameters in ((self.net, self.net.parameters()),
                                (reduced, reduced.optimization_parameters("resistance"))):
            optimizer = torch.optim.LBFGS(parameters, max_iter=3,
                                         line_search_fn="strong_wolfe")

            def closure():
                optimizer.zero_grad(set_to_none=True)
                loss = objective(net, x, t, "resistance", quadrature(16))
                loss.backward()
                return loss

            optimizer.step(closure)
        for name, parameter in self.net.state_dict().items():
            torch.testing.assert_close(parameter, reduced.state_dict()[name],
                                       rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
