"""Check physical aggregation and phase ambiguity of the observation operators."""

import unittest

import numpy as np

from i060_observation_operators import collection_density, interval_integral, segment_inventory


class ObservationTests(unittest.TestCase):
    def test_collection_uses_mass_over_volume(self):
        # Constant total flux ratio gives the same density despite varying flow.
        t = np.array([0., 1., 2.])
        q = np.array([1., 2., 3.])
        self.assertAlmostEqual(collection_density(t, q, .2*q, .25, 1.75), 1.42)
        # Linear q=1+t and j=0.2: V=4 and solid volume=0.4 over [0,2].
        self.assertAlmostEqual(collection_density(t, q, np.full(3, .2), 0, 2), 1.21)

    def test_total_segment_is_not_mobile_concentration(self):
        result = segment_inventory(np.array([0., 1.]), np.full(2, .2), np.full(2, .04), .4, 0, 1)
        # Solid volume .36*.2+.04=.112, water volume .36*.8=.288.
        self.assertAlmostEqual(result["total_solid_volume_per_area"], .112)
        self.assertAlmostEqual(result["equivalent_water_cement_ratio"], .288/(3.1*.112))

    def test_partition_cannot_be_recovered_from_total(self):
        x = np.array([0., 1.])
        mobile = segment_inventory(x, np.full(2, .2), np.zeros(2), .4, 0, 1)
        deposited = segment_inventory(x, np.zeros(2), np.full(2, .08), .4, 0, 1)
        self.assertAlmostEqual(mobile["equivalent_water_cement_ratio"], deposited["equivalent_water_cement_ratio"])
        self.assertNotEqual(mobile["deposited_solid_volume_per_area"], deposited["deposited_solid_volume_per_area"])

    def test_segment_support_is_additive(self):
        x = np.linspace(0., 1., 11)
        c, sigma = .2-.1*x, .05*(1-x)
        whole = segment_inventory(x, c, sigma, .4, 0, 1)["total_solid_volume_per_area"]
        pieces = sum(segment_inventory(x, c, sigma, .4, a, b)["total_solid_volume_per_area"]
                     for a, b in [(0, .37), (.37, 1)])
        self.assertAlmostEqual(whole, pieces)

    def test_unobserved_support_and_empty_collection_fail(self):
        with self.assertRaises(ValueError):
            interval_integral([0, 1], [0, 1], -.1, 1)
        with self.assertRaises(ValueError):
            collection_density([0, 1], [0, 0], [0, 0], 0, 1)


if __name__ == "__main__":
    unittest.main()
