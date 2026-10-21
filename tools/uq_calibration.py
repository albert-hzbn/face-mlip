"""Uncertainty calibration of the five-model flagship ensemble on the aspirin test set (CPU only).

usage: python tools/uq_calibration.py   -> runs/uq_calibration.json (reads runs/paper_preds_aspirin.npz)
Spread = sample standard deviation over members (ddof=1). Per structure: mean over atoms and components.
"""
import json, itertools
import numpy as np
from scipy.stats import spearmanr

d = np.load("runs/paper_preds_aspirin.npz")
Fe, Fr = d["F_ens"], d["F_ref"]                    # (M, S, N, 3), (S, N, 3)
M = Fe.shape[0]
err_c = (Fe.mean(0) - Fr) * 1e3                    # meV/A, per component
sd_c = Fe.std(0, ddof=1) * 1e3
sp = sd_c.mean((1, 2)); er = np.abs(err_c).mean((1, 2))
out = {"members": int(M), "estimator": "sample std (ddof=1) over members",
       "spearman_structure": float(spearmanr(sp, er)[0]),
       "mean_spread_meV_A": float(sp.mean()), "mean_abs_error_meV_A": float(er.mean())}
# binned calibration: deciles of the predicted spread, mean |error| and RMS error per bin (component level)
s, e = sd_c.ravel(), err_c.ravel()
edges = np.quantile(s, np.linspace(0, 1, 11))
bins = []
for k in range(10):
    m = (s >= edges[k]) & (s <= edges[k + 1])
    bins.append({"spread_rms": float(np.sqrt((s[m] ** 2).mean())), "error_rms": float(np.sqrt((e[m] ** 2).mean()))})
out["binned_component_deciles"] = bins
# Gaussian negative log-likelihood per component, with and without one global scale factor fitted on the test set
nll = lambda sig: float(np.mean(0.5 * np.log(2 * np.pi * sig ** 2) + e ** 2 / (2 * sig ** 2)))
a = float(np.sqrt(np.mean(e ** 2 / s ** 2)))
out["nll_component"] = nll(s); out["recalibration_factor"] = a; out["nll_component_recalibrated"] = nll(a * s)
out["fraction_within_1sigma"] = float(np.mean(np.abs(e) < s)); out["fraction_within_2sigma"] = float(np.mean(np.abs(e) < 2 * s))
# Spearman versus ensemble size (mean over all member subsets)
size = {}
for k in range(2, M + 1):
    rs = []
    for sub in itertools.combinations(range(M), k):
        f = Fe[list(sub)]
        rs.append(spearmanr(f.std(0, ddof=1).mean((1, 2)), np.abs(f.mean(0) - Fr).mean((1, 2)))[0])
    size[k] = float(np.mean(rs))
out["spearman_vs_ensemble_size"] = size
json.dump(out, open("runs/uq_calibration.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "binned_component_deciles"}, indent=1))
print("deciles (spread_rms -> error_rms):", [(round(b["spread_rms"], 1), round(b["error_rms"], 1)) for b in bins])
