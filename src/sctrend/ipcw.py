import numpy as np

def _km_weights(event_times, event_observed):
    order = np.argsort(event_times)
    t = event_times[order]
    d = event_observed[order]
    n = len(t)
    at_risk = n - np.arange(n)
    surv_prob = 1 - d / at_risk
    G = np.cumprod(surv_prob)
    G_tm = np.r_[1.0, G[:-1]]
    return order, G_tm

def concordance_index_ipcw(event_times, event_observed, lam, times, tied_tol=1e-8):
    event_times = np.asarray(event_times, float)
    event_observed = np.asarray(event_observed, int)
    lam = np.asarray(lam, float)
    times = np.asarray(times, float)
    
    delta = np.diff(times)
    cumH_tbl = np.cumsum(lam * delta, axis=1)
    order, _ = _km_weights(event_times, 1 - event_observed)
    w = event_observed[order].astype(float)

    T = event_times[order]
    L = lam[order]
    H = cumH_tbl[order]

    concord = tied = comp = 0.0
    K = len(times) - 1

    for i in range(len(T)):
        if w[i] == 0:
            continue
        mask = T > T[i] + tied_tol
        if not mask.any():
            continue

        k_star = np.searchsorted(times[1:-1], T[i], side="right")
        k_star = min(k_star, K - 1)

        if k_star == 0:
            H_prev_i = 0.0
        else:
            H_prev_i = H[i, k_star - 1]
        Lambda_i = H_prev_i + L[i, k_star] * (T[i] - times[k_star])

        if k_star == 0:
            H_prev_j = 0.0
        else:
            H_prev_j = H[mask, k_star - 1]
        Lambda_j = H_prev_j + L[mask, k_star] * (T[i] - times[k_star])

        w_i = w[i]
        concord += w_i * np.sum(Lambda_j < Lambda_i - tied_tol)
        tied += w_i * np.sum(np.abs(Lambda_j - Lambda_i) <= tied_tol)
        comp += w_i * np.sum(mask)

    c_hat = (concord + 0.5 * tied) / comp
    return c_hat, comp
