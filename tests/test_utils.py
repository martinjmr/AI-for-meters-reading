"""Unit tests for utility functions."""

import math
import unittest

import numpy as np

from src.utils import compute_angle, compute_center


class TestGeometryUtils(unittest.TestCase):
    """Test geometry calculation utilities."""

    def test_compute_center(self):
        """Test bounding box center calculation."""
        box = (0, 0, 10, 10)
        center = compute_center(box)
        self.assertEqual(center, (5, 5))

        box = (10, 20, 30, 40)
        center = compute_center(box)
        self.assertEqual(center, (20, 30))

    def test_compute_angle_horizontal(self):
        """Test angle calculation for horizontal line."""
        p1 = (0, 0)
        p2 = (10, 0)
        angle = compute_angle(p1, p2)
        self.assertAlmostEqual(angle, 0, places=5)

    def test_compute_angle_vertical(self):
        """Test angle calculation for vertical line."""
        p1 = (0, 0)
        p2 = (0, 10)
        angle = compute_angle(p1, p2)
        self.assertAlmostEqual(angle, 90, places=5)

    def test_compute_angle_45_degrees(self):
        """Test angle calculation for 45-degree line."""
        p1 = (0, 0)
        p2 = (10, 10)
        angle = compute_angle(p1, p2)
        self.assertAlmostEqual(angle, 45, places=5)


if __name__ == "__main__":
    unittest.main()
