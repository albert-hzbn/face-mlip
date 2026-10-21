"""Teacher-labelled augmentation for ensemble distillation (no new DFT data).

Rattled copies of the rMD17 *training* configurations (Gaussian displacements, sigma ~ U[s_min, s_max]) are labelled
with the mean energy/forces of an ensemble of trained FACE models. Test configurations are never touched.
usage: python tools/make_distill.py <mol> <n_per_config> <out.npz> <ckpt1> [<ckpt2> ...]
"""
import sys
import numpy as np
import torch
sys.path.insert(0, ".")
from face import safety
from face.data import load_rmd17, MoleculeSet
from face.calculator import load_face

safety.install()
mol, nper, out, paths = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4:]
s_min, s_max = 0.02, 0.06
Z, R, E, Fo, idx = load_rmd17(mol)
tr = idx["train"]
rng = np.random.default_rng(1234)
base = np.repeat(R[tr], nper, axis=0)
sig = rng.uniform(s_min, s_max, size=(len(base), 1, 1))
Raug = base + sig * rng.normal(size=base.shape)
print(f"{len(Raug)} augmented configurations from {len(tr)} training configurations", flush=True)

Es, Fs = [], []
for p in paths:
    m, c = load_face(p)
    ds = MoleculeSet(Z, Raug, np.zeros(len(Raug)), np.zeros_like(Raug), "cuda", torch.float32)
    e_all, f_all = [], []
    for k in range(0, len(Raug), 200):
        g = ds.batch(np.arange(k, min(k + 200, len(Raug))))
        o = m(g)
        e_all.append(o["energy"].detach().double().cpu().numpy() + c["Eref"])
        f_all.append(o["forces"].detach().double().cpu().numpy().reshape(-1, len(Z), 3))
    Es.append(np.concatenate(e_all)); Fs.append(np.concatenate(f_all))
    print(f"labelled with {p}", flush=True)
Es, Fs = np.array(Es), np.array(Fs)
np.savez(out, Z=Z, R=Raug, E=Es.mean(0), F=Fs.mean(0), E_std=Es.std(0), F_std=Fs.std(0).mean((1, 2)), sigma=sig.ravel())
print(f"saved {out}: teacher force spread mean {1e3 * Fs.std(0).mean():.2f} meV/A, energy spread mean {1e3 * Es.std(0).mean():.2f} meV")
