from __future__ import division, print_function
import numpy as np
from scipy import signal


def windowed_mean(x, window_len):
    """
    Get moving windowed mean of a signal (computed using fft when possible).

    :param x: (n_cells x n_timepoints) array
    :param window_len: number of timepoints to use in averaging window
    :return:
    """
    rect_filter = np.ones((window_len,), dtype=float) / window_len
    means = np.nan * np.ones(x.shape)

    for ctr, trace in enumerate(x):
        # calculate middle portion of convolution
        middle = signal.fftconvolve(trace, rect_filter, 'valid')
        # calculate the beginning and end
        n_pts = int(np.floor(window_len / 2))
        beginning = (trace[:window_len-1].cumsum() / np.arange(1, window_len))[-n_pts:]
        if window_len % 2 == 1:
            ending = (trace[-window_len+1:][::-1].cumsum() / np.arange(1, window_len))[-n_pts:][::-1]
        else:
            ending = (trace[-window_len+1:][::-1].cumsum() / np.arange(1, window_len))[-n_pts+1:][::-1]

        # store signal
        means[ctr] = np.concatenate([beginning, middle, ending])

    return means


def get_chunks(x, starts, ends, axis):
    """
    Break up a multi-dimensional time-series into chunks along a given axis.

    :param x: time-series (numpy array)
    :param starts: indices of chunk starts
    :param ends: indices of chunk ends
    :param axis: which axis to break time-series up along
    :return: list of chunks, each of which is itself a time-series.
    """

    chunks = []
    for start, end in zip(starts, ends):
        # TODO: make this work with arbitrary axes for arbitrarily dimensioned arrays
        if axis == 0:
            chunks.append(x[start:end, :])
        elif axis == 1:
            chunks.append(x[:, start:end])

    return chunks