"""Ensemble of FACE checkpoints on the rMD17 test split: mean prediction errors + uncertainty calibration."""
import sys, json
import numpy as np
import torch
sys.path.insert(0, ".")
from face import safety
from face.data import load_rmd17, MoleculeSet
from face.calculator import load_face

safety.install()
mol, paths = sys.argv[1], sys.argv[2:]
Z, R, E, Fo, idx = load_rmd17(mol)
te = idx["test"]
preds = []
for p in paths:
    m, c = load_face(p)
    ds = MoleculeSet(Z, R, E - c["Eref"], Fo, "cuda", torch.float32)
    Es, Fs = [], []
    for k in range(0, len(te), 50):
        g = ds.batch(te[k:k + 50]); o = m(g)
        Es.append(o["energy"].detach().double().cpu().numpy() + c["Eref"])
        Fs.append(o["forces"].detach().double().cpu().numpy().reshape(-1, len(Z), 3))
    preds.append((np.concatenate(Es), np.concatenate(Fs)))
Eref, Fref = E[te], Fo[te]
for k, (e, f) in enumerate(preds):
    print(f"member {k} ({paths[k]}): E {1e3*np.abs(e-Eref).mean():.3f}  F {1e3*np.abs(f-Fref).mean():.3f}")
Em = np.mean([p[0] for p in preds], 0); Fm = np.mean([p[1] for p in preds], 0)
print(f"ENSEMBLE of {len(preds)}: E {1e3*np.abs(Em-Eref).mean():.3f} meV  F {1e3*np.abs(Fm-Fref).mean():.3f} meV/A")
# uncertainty: per-structure force spread vs per-structure force error
spread = np.std([p[1] for p in preds], 0).mean((1, 2)); err = np.abs(Fm - Fref).mean((1, 2))
print(f"Spearman(spread, error) per structure: {np.corrcoef(np.argsort(np.argsort(spread)), np.argsort(np.argsort(err)))[0,1]:.3f}")
