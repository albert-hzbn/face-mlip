"""Compare the interatomic-distance ensemble of FACE Langevin MD with the rMD17 reference (aspirin).
usage: python tools/md_structure.py  (reads runs/paper_langevin_aspirin.npz)"""
import sys
import numpy as np
sys.path.insert(0, ".")
from face.data import load_rmd17

lgv = np.load("runs/paper_langevin_aspirin.npz")
Z, R, *_ = load_rmd17("aspirin")
n = len(Z); iu = np.triu_indices(n, 1)
burn = lgv["t_ps"] > 1.0
dmd = np.linalg.norm(lgv["pos"][burn][:, :, None] - lgv["pos"][burn][:, None], axis=-1)[:, iu[0], iu[1]]
sel = np.random.default_rng(0).choice(len(R), 5000, replace=False)
dref = np.linalg.norm(R[sel][:, :, None] - R[sel][:, None], axis=-1)[:, iu[0], iu[1]]
bins = np.linspace(0.8, 7.5, 271)
hr, _ = np.histogram(dref.ravel(), bins, density=True); hm, _ = np.histogram(dmd.ravel(), bins, density=True)
print("mean T after 1 ps: %.1f K" % lgv["T"][burn].mean())
print("histogram overlap: %.3f" % (np.minimum(hr, hm).sum() * (bins[1] - bins[0])))
bonded = np.linalg.norm(R[0][:, None] - R[0][None], axis=-1)[iu] < 1.6
for a, b, name in [(6, 1, "C-H"), (6, 6, "C-C"), (6, 8, "C-O"), (8, 1, "O-H")]:
    m = bonded & (((Z[iu[0]] == a) & (Z[iu[1]] == b)) | ((Z[iu[0]] == b) & (Z[iu[1]] == a)))
    print("%s (%d bonds): MD %.4f +- %.4f  ref %.4f +- %.4f  diff %.4f A" % (
        name, m.sum(), dmd[:, m].mean(), dmd[:, m].std(), dref[:, m].mean(), dref[:, m].std(),
        dmd[:, m].mean() - dref[:, m].mean()))
