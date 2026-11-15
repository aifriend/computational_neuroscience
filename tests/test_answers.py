"""Regression tests for the worked answers in this repository.

The threshold values 5.83 and 5.978 are the expected values from the Wave 2
plan. The script solve_likelihood_ratio_final.py still reports 5.667, so these
tests describe the intended answers, not what that script prints today.
"""

import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.stats import norm

from compute_sta import compute_sta

MU1, SIGMA1 = 5.0, 0.5
MU2, SIGMA2 = 7.0, 1.0


def likelihood_ratio(r):
    return norm.pdf(r, MU1, SIGMA1) / norm.pdf(r, MU2, SIGMA2)


def threshold_for_ratio(target):
    """Find r between the two means where P(r|s1)/P(r|s2) equals target."""
    return brentq(lambda r: likelihood_ratio(r) - target, MU1, MU2)


def test_equal_cost_threshold_is_5_83():
    assert threshold_for_ratio(1.0) == pytest.approx(5.83, abs=5e-3)


def test_asymmetric_cost_threshold_is_5_978():
    assert threshold_for_ratio(0.5) == pytest.approx(5.978, abs=5e-3)


def test_map_posterior_of_disease_is_about_4_95e_minus_7():
    prior = 1e-8
    sensitivity = 0.99
    false_positive = 0.02
    evidence = sensitivity * prior + false_positive * (1 - prior)
    posterior = sensitivity * prior / evidence
    assert posterior == pytest.approx(4.95e-7, rel=1e-2)
    # MAP decides "no disease" while ML (0.99 > 0.02) decides "disease".
    assert posterior < 1 - posterior


def test_sta_recovers_template_on_synthetic_spike_train():
    n = 10
    template = np.arange(1, n + 1, dtype=float)
    stim = np.zeros(500)
    rho = np.zeros(500, dtype=int)
    for t in (50, 120, 200, 310, 400):
        rho[t] = 1
        stim[t - n + 1:t + 1] = template  # window ends at the spike sample
    sta = compute_sta(stim, rho, n)
    assert sta.shape == (n,)
    np.testing.assert_allclose(sta, template)


def test_sta_ignores_spikes_before_first_full_window():
    n = 10
    stim = np.ones(100)
    rho = np.zeros(100, dtype=int)
    rho[3] = 1  # too early: not enough history
    assert np.all(compute_sta(stim, rho, n) == 0)
