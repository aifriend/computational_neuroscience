"""
Created on Wed Apr 22 15:21:11 2015

Code to compute spike-triggered average.
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


def compute_sta(stim, rho, num_timesteps):
    """Compute the spike-triggered average from a stimulus and spike-train.
    
    Args:
        stim: stimulus time-series
        rho: spike-train time-series
        num_timesteps: how many timesteps to use in STA
        
    Returns:
        spike-triggered average for num_timesteps timesteps before spike"""
    
    sta = np.zeros((num_timesteps,))

    # This command finds the indices of all of the spikes that occur
    # after 300 ms into the recording.
    spike_times = rho[num_timesteps:].nonzero()[0] + num_timesteps

    # Fill in this value. Note that you should not count spikes that occur
    # before 300 ms into the recording.
    num_spikes = len(spike_times)
    
    # Compute the spike-triggered average using Bayes' rule.
    # Bayes' rule: P(stimulus | spike) ∝ P(spike | stimulus) * P(stimulus)
    # The STA is E[stimulus | spike], computed by averaging all stimulus
    # segments that precede spikes.
    # 
    # For each spike, extract the stimulus window from num_timesteps before
    # the spike (exclusive) to the spike time (inclusive).
    for spike_time in spike_times:
        # Extract stimulus segment: from (spike_time - num_timesteps) to spike_time
        # This gives us num_timesteps samples before the spike
        stimulus_window = stim[spike_time - num_timesteps:spike_time]
        sta += stimulus_window
    
    # Average by dividing by the number of spikes (Bayes' rule: conditional expectation)
    if num_spikes > 0:
        sta = sta / num_spikes
    
    return sta