from __future__ import division
import unittest
import numpy as np
import fluorescence as fl


class BasicFunctionsTestCase(unittest.TestCase):

    def test_windowed_mean_odd_window_len(self):

        x = np.array([[0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]], dtype=float)
        window_len = 5
        x_normed_correct = np.array([[1, 1.5, 2, 3, 18/5, 19/5, 18/5, 3, 2, 1.5, 1]])

        x_normed = fl.windowed_mean(x, window_len)
        self.assertEqual(len(x_normed), len(x_normed_correct))
        np.testing.assert_array_almost_equal(x_normed, x_normed_correct)

    def test_windowed_mean_even_window_len(self):

        x = np.array([[0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]], dtype=float)
        window_len = 4
        x_normed_correct = np.array([[0.5, 1, 1.5, 2.5, 3.5, 4, 4, 3.5, 2.5, 1.5, 1]])

        x_normed = fl.windowed_mean(x, window_len)
        self.assertEqual(len(x_normed), len(x_normed_correct))
        np.testing.assert_array_almost_equal(x_normed, x_normed_correct)


if __name__ == '__main__':
    unittest.main()