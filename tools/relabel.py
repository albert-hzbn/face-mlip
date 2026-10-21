"""Relabel an existing distillation set (configurations unchanged) with a new teacher ensemble.
usage: python tools/relabel.py <in.npz> <out.npz> <ckpt1> [<ckpt2> ...]"""
import sys
import numpy as np
import torch
sys.path.insert(0, ".")
from face import safety
from face.data import MoleculeSet
from face.calculator import load_face

safety.install()
src, out, paths = sys.argv[1], sys.argv[2], sys.argv[3:]
d = np.load(src)
Z, R = d["Z"], d["R"]
Es, Fs = [], []
for p in paths:
    m, c = load_face(p)
    ds = MoleculeSet(Z, R, np.zeros(len(R)), np.zeros_like(R), "cuda", torch.float32)
    e, f = [], []
    for k in range(0, len(R), 200):
        o = m(ds.batch(np.arange(k, min(k + 200, len(R)))))
        e.append(o["energy"].detach().double().cpu().numpy() + c["Eref"])
        f.append(o["forces"].detach().double().cpu().numpy().reshape(-1, len(Z), 3))
    Es.append(np.concatenate(e)); Fs.append(np.concatenate(f))
Es, Fs = np.array(Es), np.array(Fs)
np.savez(out, Z=Z, R=R, E=Es.mean(0), F=Fs.mean(0), F_std=Fs.std(0).mean((1, 2)))
print(f"relabelled {len(R)} configurations with {len(paths)} models; mean force spread {1e3 * Fs.std(0).mean():.2f} meV/A")
