from __future__ import division
import unittest
import numpy as np
import time_series


class BasicFunctionsTestCase(unittest.TestCase):

    def test_windowed_mean_odd_window_len(self):

        x = np.array([[0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]], dtype=float)
        window_len = 5
        x_normed_correct = np.array([[1, 1.5, 2, 3, 18/5, 19/5, 18/5, 3, 2, 1.5, 1]])

        x_normed = time_series.windowed_mean(x, window_len)
        self.assertEqual(len(x_normed), len(x_normed_correct))
        np.testing.assert_array_almost_equal(x_normed, x_normed_correct)

    def test_windowed_mean_even_window_len(self):

        x = np.array([[0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]], dtype=float)
        window_len = 4
        x_normed_correct = np.array([[0.5, 1, 1.5, 2.5, 3.5, 4, 4, 3.5, 2.5, 1.5, 1]])

        x_normed = time_series.windowed_mean(x, window_len)
        self.assertEqual(len(x_normed), len(x_normed_correct))
        np.testing.assert_array_almost_equal(x_normed, x_normed_correct)

    def test_get_chunks(self):

        x = np.random.normal(0, 1, (100, 80))
        starts = [20, 50]
        ends = [30, 70]

        # test dimension 0
        chunks = time_series.get_chunks(x, starts, ends, axis=0)
        self.assertEqual(len(chunks), 2)
        for chunk, start, end in zip(chunks, starts, ends):
            np.testing.assert_array_almost_equal(chunk, x[start:end, :])

        # test dimension 1
        chunks = time_series.get_chunks(x, starts, ends, axis=1)
        self.assertEqual(len(chunks), 2)
        for chunk, start, end in zip(chunks, starts, ends):
            np.testing.assert_array_almost_equal(chunk, x[:, start:end])

    def test_subtract_first(self):

        x = np.array([[1, 2, 3, 4],
                      [2, 3, 4, 5],
                      [6, 7, 1, 4]])

        # test dimension 0
        x_correct = np.array([[0, 0, 0, 0],
                              [1, 1, 1, 1],
                              [5, 5, -2, 0]])
        np.testing.assert_array_almost_equal(time_series.subtract_first(x, axis=0), x_correct)

        # test dimension 1
        x_correct = np.array([[0, 1, 2, 3],
                              [0, 1, 2, 3],
                              [0, 1, -5, -2]])
        np.testing.assert_array_almost_equal(time_series.subtract_first(x, axis=1), x_correct)


if __name__ == '__main__':
    unittest.main()