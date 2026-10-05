"""Tests for task-time interpolation and the quantities accumulated up to it."""

import unittest

import numpy as np

from analyze_i060_supply_tasks import evaluate_task, first_attainment


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

    def test_unattained_target_is_not_clipped_to_the_final_time(self):
        with self.assertRaisesRegex(ValueError, "not reached"):
            first_attainment(np.array([0., 1.]), np.array([0., .02]), .03)


if __name__ == "__main__":
    unittest.main()
