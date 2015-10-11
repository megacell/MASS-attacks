import unittest
import numpy as np

from utils import simplex_projection, is_equal

class TestUtils(unittest.TestCase):

    def test_simplex_projection(self):
        v = np.array([1, 0, 0, 0, 0, 0])
        self.assertTrue(is_equal(simplex_projection(v), v))

        v = np.array([0.1, 0.2, 0.3, 0.4])
        self.assertTrue(is_equal(simplex_projection(v), v))

        v = np.array([10, 10])
        w = np.array([0.5, 0.5])
        self.assertTrue(is_equal(simplex_projection(v), w))

        v = np.array([-10, -10])
        w = np.array([0.5, 0.5])
        self.assertTrue(is_equal(simplex_projection(v), w))

        v = np.array([-10, 10])
        w = np.array([0, 1])
        self.assertTrue(is_equal(simplex_projection(v), w))

        for _ in range(20):
            v = (np.random.rand(20) - np.ones(20) * 0.5) * 100
            self.assertTrue(is_equal(np.sum(simplex_projection(v)), 1))
            self.assertTrue(is_equal(np.sum(simplex_projection(v, 5)), 5))
