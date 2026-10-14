"""Check cohort flux, pore-volume signs and independent source accounting."""

import unittest

import numpy as np
import torch

from solve_i060_origins import Labels, START, inventory_residual, residual


class OriginTests(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(7)

    def test_nonnegative_labels_and_separate_feed_fluxes(self):
        net = Labels(12, .14, .01, 1.)
        t = torch.linspace(0, 1, 23)[:, None]
        q, feed = .4+0*t, .06+0*t
        c, j = net(torch.cat((0*t, t), 1), q, feed)
        torch.testing.assert_close(j[:, 0], torch.zeros(len(t)))
        torch.testing.assert_close(j[:, 1:], q*feed)
        self.assertTrue(torch.all(c >= 0))
        self.assertEqual(c[0, 1].item(), 0.)
        c, j = net(torch.cat((torch.ones_like(t), t), 1), q, feed)
        torch.testing.assert_close(j, q*c)

    def test_mass_operator_uses_physical_time_and_pore_volume_change(self):
        class LinearFields:
            cin, dispersion, end = .14, .01, 1.

            def __call__(self, xt, q, feed):
                x, u = xt[:, :1], xt[:, 1:]
                return torch.cat((.02+.03*u+.01*x, .04*u+.02*x), 1), torch.cat((.05*x, .06*x), 1)

        data = {"xt": torch.tensor([[.2, .3], [.6, .7]]),
                "q": torch.full((2, 1), .4), "phi": torch.full((2, 1), .3),
                "ds": torch.full((2, 1), .02), "capture": torch.full((2, 1), 1.1),
                "feed": torch.full((2, 1), .06)}
        net = LinearFields()
        c, j = net(data["xt"], data["q"], data["feed"])
        expected_mass = (.3*torch.tensor([.03, .04])/(1-START)
                         -.02*c+torch.tensor([.05, .06])+1.1*c)/.14
        expected_flux = (j-.4*c+.3*.01*torch.tensor([.01, .02]))/(.14*.1)
        mass, flux = residual(net, data)
        torch.testing.assert_close(mass, expected_mass)
        torch.testing.assert_close(flux, expected_flux)

    def test_warm_start_preserves_weights_when_positive_map_is_applied(self):
        initial = Labels(12, .14, .01, 1., positive=False)
        final = Labels(12, .14, .01, 1.)
        final.load_state_dict(initial.state_dict())
        raw = torch.tensor([[-.1, .2], [.3, -.4]])
        u = torch.tensor([[0.], [.5]])
        torch.testing.assert_close(initial.concentrations(raw, u), .14*raw)
        self.assertTrue(torch.all(final.concentrations(raw, u) >= 0))
        self.assertEqual(final.concentrations(raw, u)[0, 1].item(), 0.)

    def test_balance_uses_actual_initial_mass_and_zero_new_inventory(self):
        class ConstantFields:
            def __call__(self, xt, q, feed):
                return torch.tensor([[.2, .05]]).expand(len(xt), -1), torch.zeros(len(xt), 2)

        nt = 3
        data = {"xt": torch.zeros(6*nt, 2), "q": torch.zeros(6*nt, 1),
                "feed": torch.zeros(6*nt, 1), "phi": torch.full((6*nt, 1), .4),
                "c": torch.full((6*nt, 1), .2), "capture": torch.zeros(6*nt, 1)}
        face = {"xt": torch.zeros(7*nt, 2), "q": torch.zeros(7*nt, 1), "feed": torch.zeros(7*nt, 1)}
        value = inventory_residual(ConstantFields(), data, face, torch.tensor([1/6]), torch.linspace(START, 1, nt))
        np.testing.assert_allclose(value[:, :, 0], 0., atol=1e-12)
        np.testing.assert_allclose(value[:, :, 1], 2.)


if __name__ == "__main__":
    unittest.main()
