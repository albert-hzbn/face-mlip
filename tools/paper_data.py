"""Data for the manuscript figures that need fresh evaluations (written to runs/paper_*.npz).

usage: python tools/paper_data.py preds | md | langevin
  preds    per-structure test predictions: best single model + five-seed flagship ensemble (aspirin)
  md       NVE trajectory (velocity Verlet, 0.5 fs, 20 ps, T0 = 500 K) with the best single model
  langevin Langevin MD at 500 K (0.5 fs, 20 ps) with the best single model -> interatomic-distance samples
"""
import sys, time
import numpy as np
import torch
sys.path.insert(0, ".")
from face import safety
from face.data import load_rmd17, MoleculeSet
from face.calculator import load_face, FACECalculator

BEST = "runs/gen4_eft_aspirin.pt"
ENSEMBLE = [f"runs/{t}_aspirin.pt" for t in
            ("long_flag", "long_flag_s1", "long_flag_ep", "long_flag_ep_s2", "long_flag_ep_s3")]


def predict(path, Z, R, E, Fo, te):
    m, c = load_face(path)
    ds = MoleculeSet(Z, R, E - c["Eref"], Fo, "cuda", torch.float32)
    Es, Fs = [], []
    for k in range(0, len(te), 50):
        o = m(ds.batch(te[k:k + 50]))
        Es.append(o["energy"].detach().double().cpu().numpy() + c["Eref"])
        Fs.append(o["forces"].detach().double().cpu().numpy().reshape(-1, len(Z), 3))
    return np.concatenate(Es), np.concatenate(Fs)


def preds():
    Z, R, E, Fo, idx = load_rmd17("aspirin")
    te = idx["test"]
    out = {"Z": Z, "E_ref": E[te], "F_ref": Fo[te]}
    out["E_best"], out["F_best"] = predict(BEST, Z, R, E, Fo, te)
    ens = [predict(p, Z, R, E, Fo, te) for p in ENSEMBLE]
    out["E_ens"] = np.stack([e for e, _ in ens]); out["F_ens"] = np.stack([f for _, f in ens])
    np.savez_compressed("runs/paper_preds_aspirin.npz", **out)
    print("best   E %.3f F %.3f" % (1e3 * np.abs(out["E_best"] - out["E_ref"]).mean(),
                                    1e3 * np.abs(out["F_best"] - out["F_ref"]).mean()))
    print("ens5   E %.3f F %.3f" % (1e3 * np.abs(out["E_ens"].mean(0) - out["E_ref"]).mean(),
                                    1e3 * np.abs(out["F_ens"].mean(0) - out["F_ref"]).mean()))


def md(langevin=False):
    from ase import Atoms, units
    from ase.md.verlet import VelocityVerlet
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
    Z, R, *_ = load_rmd17("aspirin")
    at = Atoms(numbers=Z, positions=R[0]); at.calc = FACECalculator(BEST)
    rng = np.random.default_rng(0)
    MaxwellBoltzmannDistribution(at, temperature_K=500, rng=rng); Stationary(at); ZeroRotation(at)
    if langevin:
        from ase.constraints import FixCom
        at.set_constraint(FixCom())   # fixcm=True mis-samples small systems (ASE >= 3.28 warning)
        dyn = Langevin(at, 0.5 * units.fs, temperature_K=500, friction=0.01 / units.fs, rng=rng, fixcm=False)
        nsteps, every = 40000, 20
    else:
        dyn = VelocityVerlet(at, timestep=0.5 * units.fs)
        nsteps, every = 40000, 10
    rec = {"t_ps": [], "Epot": [], "Ekin": [], "T": [], "pos": []}
    t0 = time.time()
    for k in range(0, nsteps, every):
        dyn.run(every)
        rec["t_ps"].append((k + every) * 0.5e-3)
        rec["Epot"].append(at.get_potential_energy()); rec["Ekin"].append(at.get_kinetic_energy())
        rec["T"].append(at.get_temperature())
        if langevin:
            rec["pos"].append(at.get_positions().copy())
    out = {k: np.array(v) for k, v in rec.items() if len(v)}
    out["Z"] = Z
    name = "runs/paper_langevin_aspirin.npz" if langevin else "runs/paper_nve_aspirin.npz"
    np.savez_compressed(name, **out)
    print(name, "wall %.0f s" % (time.time() - t0), "mean T %.1f" % out["T"].mean())


if __name__ == "__main__":
    safety.install()
    torch.set_default_dtype(torch.float32)
    {"preds": preds, "md": lambda: md(False), "langevin": lambda: md(True)}[sys.argv[1]]()
