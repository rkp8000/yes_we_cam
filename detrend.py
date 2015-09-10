from __future__ import division, print_function
import numpy as np
from scipy import signal


def windowed_mean(traces, window_len):
    """

    :param traces: (n_cells x n_timepoints) array
    :param window_len: number of timepoints to use in averaging window
    :return:
    """
    rect_filter = np.ones((window_len,), dtype=float) / window_len
    new_traces = np.nan * np.ones(traces.shape)

    for trace_ctr, trace in enumerate(traces):
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
        new_traces[trace_ctr] = np.concatenate([beginning, middle, ending])

    return new_traces