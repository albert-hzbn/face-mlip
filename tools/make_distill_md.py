"""Ensemble-distillation data from teacher MD (no new DFT data).

All rMD17 *training* configurations are propagated simultaneously (one batched graph) with Langevin dynamics at
temperature T using the ensemble-mean forces of trained FACE models; snapshots every `every` steps are labelled with
the ensemble mean energy/forces. Snapshots whose ensemble force spread exceeds `max_spread` (an uncertainty gate,
calibrated against the test-set spread) are discarded. Test configurations are never used.
usage: python tools/make_distill_md.py <mol> <T_K> <n_steps> <every> <max_spread_meV_A> <out.npz> <ckpt1> [...]
"""
import sys
import numpy as np
import torch
from ase.data import atomic_masses
sys.path.insert(0, ".")
from face import safety
from face.data import load_rmd17, MoleculeSet
from face.calculator import load_face

safety.install()
mol, T, nsteps, every, max_spread, out = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), \
    float(sys.argv[5]) / 1000, sys.argv[6]
paths = sys.argv[7:]
Z, R, E, Fo, idx = load_rmd17(mol)
tr = idx["train"]
models = [load_face(p) for p in paths]
n, B = len(Z), len(tr)
dev = "cuda"
kB = 8.617333e-5                      # eV/K
mass = torch.tensor(atomic_masses[Z], device=dev, dtype=torch.float64)[None, :, None]   # amu
conv = 9.648533e-3                    # eV/A/amu -> A/fs^2
dt, gamma = 0.5, 0.01                 # fs, 1/fs (Langevin friction)
x = torch.tensor(R[tr], device=dev, dtype=torch.float64)
import os
gen = torch.Generator(device=dev).manual_seed(int(os.environ.get("MD_SEED", 0)))
v = torch.randn(x.shape, device=dev, dtype=torch.float64, generator=gen) * torch.sqrt(kB * T / mass * conv)


def ensemble(xx, chunk=190):
    ds = MoleculeSet(Z, xx.cpu().numpy(), np.zeros(B), np.zeros((B, n, 3)), dev, torch.float32)
    Es, Fs = [], []
    for m, c in models:
        e_m, f_m = [], []
        for k in range(0, B, chunk):
            o = m(ds.batch(np.arange(k, min(k + chunk, B))))
            e_m.append(o["energy"].detach().double() + c["Eref"])
            f_m.append(o["forces"].detach().double().view(-1, n, 3))
        Es.append(torch.cat(e_m)); Fs.append(torch.cat(f_m))
    Es, Fs = torch.stack(Es), torch.stack(Fs)
    return Es.mean(0), Fs.mean(0), Fs.std(0).mean((1, 2))


e, f, sp = ensemble(x)
keepR, keepE, keepF, keepS, dropped = [], [], [], [], 0
c1 = np.exp(-gamma * dt)
c2 = torch.sqrt((1 - c1 ** 2) * kB * T / mass * conv)
for step in range(1, nsteps + 1):
    # BAOAB Langevin
    v = v + 0.5 * dt * f / mass * conv
    x = x + 0.5 * dt * v
    v = c1 * v + c2 * torch.randn(x.shape, device=dev, dtype=torch.float64, generator=gen)
    x = x + 0.5 * dt * v
    e, f, sp = ensemble(x)
    v = v + 0.5 * dt * f / mass * conv
    if step % every == 0:
        ok = sp < max_spread
        dropped += int((~ok).sum())
        keepR.append(x[ok].cpu().numpy()); keepE.append(e[ok].cpu().numpy())
        keepF.append(f[ok].cpu().numpy()); keepS.append(sp[ok].cpu().numpy())
        Tinst = (mass * v ** 2).sum((1, 2)) / conv / (3 * n * kB)
        print(f"step {step}: kept {int(ok.sum())}/{B}, mean spread {1e3 * sp.mean().item():.1f} meV/A, "
              f"T {Tinst.mean().item():.0f} K", flush=True)
np.savez(out, Z=Z, R=np.concatenate(keepR), E=np.concatenate(keepE), F=np.concatenate(keepF),
         F_std=np.concatenate(keepS))
print(f"saved {out}: {sum(len(a) for a in keepE)} snapshots, {dropped} rejected by the uncertainty gate")
