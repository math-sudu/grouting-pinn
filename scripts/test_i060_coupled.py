"""Verify state support, conserved storage and hydraulic feedback gradients."""

import math
import unittest

import torch

from i060_coupled import Fields, Problem, grad, quadrature, resistance_q, residuals


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
            self.net.state[-1].bias[0] = math.log(.08/.92)-math.log(.02/.98)
        c, _, _ = self.net(torch.tensor([[1., .9], [0., .9]]))
        self.assertAlmostEqual(float(c[0].detach()), .08, places=9)
        self.assertAlmostEqual(float(c[1].detach()), .02, places=9)

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
        parameter = self.net.state[-1].bias
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


if __name__ == "__main__":
    unittest.main()
