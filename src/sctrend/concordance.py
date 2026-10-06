"""Concordance metrics used by scTREND.

The metric implemented here is deliberately named for what it computes. It
does not estimate the censoring distribution and therefore is not an IPCW/Uno
concordance estimator.
"""

from __future__ import annotations

import numpy as np


def concordance_index_unweighted_dynamic(
    event_times,
    event_observed,
    lam,
    times,
    tied_tol=1e-8,
):
    """Return unweighted pairwise concordance for dynamic hazard predictions.

    For every observed event, the subject's cumulative hazard at its event
    time is compared with subjects whose observed time is later. Comparable
    pairs receive equal weight; no inverse-probability-of-censoring weights are
    estimated. With one time bin, this reduces to a Harrell-like unweighted
    C-index because cumulative-hazard ordering equals hazard ordering.

    Parameters are kept compatible with the historical
    concordance_index_ipcw function so existing callers can migrate without
    changing numerical behavior.
    """
    event_times = np.asarray(event_times, float) #0-5に正規化した時間を入れている
    event_observed = np.asarray(event_observed, int)
    lam = np.asarray(lam, float)
    times = np.asarray(times, float) #edgeを渡す

    delta = np.diff(times)
    cum_hazard_table = np.cumsum(lam * delta, axis=1)
    order = np.argsort(event_times)

    observed_event = event_observed[order].astype(float)
    ordered_times = event_times[order]
    ordered_hazards = lam[order]
    ordered_cumulative_hazards = cum_hazard_table[order]

    concordant = tied = comparable = 0.0
    n_time_bins = len(times) - 1

    for event_index in range(len(ordered_times)):
        if observed_event[event_index] == 0:
            continue
        later = ordered_times > ordered_times[event_index] + tied_tol
        if not later.any():
            continue

        time_bin = np.searchsorted(
            times[1:-1], ordered_times[event_index], side="right"
        )
        time_bin = min(time_bin, n_time_bins - 1)

        if time_bin == 0:
            previous_event_hazard = 0.0
            previous_later_hazard = 0.0
        else:
            previous_event_hazard = ordered_cumulative_hazards[
                event_index, time_bin - 1
            ]
            previous_later_hazard = ordered_cumulative_hazards[
                later, time_bin - 1
            ]

        event_cumulative_hazard = (
            previous_event_hazard
            + ordered_hazards[event_index, time_bin]
            * (ordered_times[event_index] - times[time_bin])
        )
        later_cumulative_hazard = (
            previous_later_hazard
            + ordered_hazards[later, time_bin]
            * (ordered_times[event_index] - times[time_bin])
        )

        pair_weight = observed_event[event_index]
        concordant += pair_weight * np.sum(
            later_cumulative_hazard < event_cumulative_hazard - tied_tol
        )
        tied += pair_weight * np.sum(
            np.abs(later_cumulative_hazard - event_cumulative_hazard) <= tied_tol
        )
        comparable += pair_weight * np.sum(later)

    c_index = (concordant + 0.5 * tied) / comparable
    return c_index, comparable
