"""Tests for task-time interpolation and the quantities accumulated up to it."""

import unittest

import numpy as np

from analyze_i060_supply_tasks import evaluate_task, first_attainment, regional_capture


class SupplyTaskTests(unittest.TestCase):
    def test_partial_interval_flux_and_spatial_inventory(self):
        times = np.array([0., 1., 3.])
        x = np.array([0., .5, 1.])
        fields = {
            "x": x, "times": times,
            "deposited": .02 * times,
            "sigma": np.tile(.02 * times, (len(x), 1)),
            "j": np.array([2 + 2 * times, 1 + times, times]),
            "q": (1 - .1 * times).reshape(-1, 1),
            "mobile": .05 * times,
        }
        case = evaluate_task(fields, .03)
        expected = {
            "time": 1.5, "deposited_inventory": .03,
            "cumulative_inlet_solids": 5.25, "cumulative_outlet_solids": 1.125,
            "cumulative_mixture_volume": 1.3875, "flow": .85,
            "mobile_inventory": .075, "half_inventory_position": .5,
            "inlet_third_deposited_fraction": 1 / 3,
        }
        for quantity, value in expected.items():
            with self.subTest(quantity=quantity):
                self.assertAlmostEqual(case[quantity], value)

    def test_first_crossing_is_used_if_inventory_later_falls(self):
        time, *_ = first_attainment(np.arange(4.), np.array([0., .04, .02, .05]), .03)
        self.assertAlmostEqual(time, .75)

    def test_regional_capture_includes_partial_space_and_time_intervals(self):
        times = np.array([0., .4, 1.])
        sigma = np.tile(.02*times, (3, 1))
        fields = {"x": np.array([0., .5, 1.]), "times": times,
                  "sigma": sigma, "c": np.full_like(sigma, .005),
                  "q": 1/(1-.02*times/.2)}
        result = regional_capture(fields, {"capacity": .2, "retention": 4.},
                                  (.25, 1.), .3, .8)
        self.assertAlmostEqual(result["deposited_gain"], .0075)
        self.assertAlmostEqual(result["integrated_capture"], .0075)

    def test_unattained_target_is_not_clipped_to_the_final_time(self):
        with self.assertRaisesRegex(ValueError, "not reached"):
            first_attainment(np.array([0., 1.]), np.array([0., .02]), .03)

    def test_regional_target_preserves_whole_column_accounting(self):
        times = np.array([0., 1., 3.])
        fields = {
            "x": np.array([0., .5, 1.]), "times": times,
            "deposited": .02*times, "sigma": np.tile(.02*times, (3, 1)),
            "j": np.array([2+2*times, 1+times, times]),
            "q": (1-.1*times).reshape(-1, 1), "mobile": .05*times,
        }
        result = evaluate_task(fields, .03, region=(.25, 1.))
        self.assertAlmostEqual(result["time"], 2.)
        self.assertAlmostEqual(result["regional_deposited_inventory"], .03)
        self.assertAlmostEqual(result["deposited_inventory"], .04)
        self.assertAlmostEqual(result["outside_deposited_inventory"], .01)
        self.assertAlmostEqual(result["mass_balance_defect"], 5.86)
        # The whole column attains 0.046, but this shorter interval does not.
        with self.assertRaisesRegex(ValueError, "not reached"):
            evaluate_task(fields, .046, region=(.25, 1.))


if __name__ == "__main__":
    unittest.main()
