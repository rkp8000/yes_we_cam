"""
Metrics for experiments. To be possibly split into multiple files should it grow too large.
"""
from __future__ import division, print_function
import numpy as np

import api
import time_series

NEURONS_PER_STIM = 10


def mean_variability_stim_conditioned(nwb_file, detrend_window):
    """
    Calculate the variability of neural responses conditioned on a specific stimulus, averaged over
    basically everything, for a single experiment.
    :param nwb_file: path to experiment's .nwb file
    :param detrend_window: window length (number of timesteps) to use when detrending fluorescence traces
    :return scalar metric quantifying stim-conditioned variability
    """
    print(nwb_file)
    # get and detrend fluorescence traces
    _, traces = api.get_fluorescence_traces(nwb_file)
    traces -= time_series.windowed_mean(traces, detrend_window)
    traces /= np.tile(traces.std(axis=1)[:, None], (1, traces.shape[1]))

    # get stimulus data
    stim_data = api.get_stimulus_table(nwb_file)

    # get unique stimulus conditions as list of tuples
    stim_conditions_df = api.get_stimulus_conditions(nwb_file, include_blanks=False)
    stim_conditions = [tuple(cond) for cond in stim_conditions_df.values]

    time_averaged_stds_stim = []
    for tf, ori in stim_conditions:
        # get all start and end times for this stimulus
        mask = (stim_data['temporal_frequency'] == tf) & (stim_data['orientation'] == ori)
        starts, ends = stim_data[['start', 'end']][mask].values.T
        min_dur = (ends - starts).min()
        diffs = ends - starts - min_dur
        ends[diffs > 0] -= diffs[diffs > 0]

        # get all neural responses to all repetitions of this stimulus
        responses_all = time_series.get_chunks(traces, starts, ends, axis=1)

        # find most responsive neurons
        responses_all = [time_series.subtract_first(response, axis=1) for response in responses_all]
        responsivenesses_trial = [np.abs(response).max(axis=1)[:, None] for response in responses_all]
        responsivenesses_stim = np.concatenate(responsivenesses_trial, axis=1).mean(axis=1)
        most_active_neurons = np.argsort(responsivenesses_stim)[-NEURONS_PER_STIM:]

        # get responses of active neurons only
        responses = [response[most_active_neurons, :] for response in responses_all]

        time_averaged_stds_neuron = []
        for ctr in range(NEURONS_PER_STIM):
            std = np.concatenate([response[ctr, :][None, :] for response in responses], axis=0).std(axis=0)
            time_averaged_stds_neuron.append(std.mean())

        time_averaged_stds_stim.append(np.mean(time_averaged_stds_neuron))

    return np.mean(time_averaged_stds_stim)