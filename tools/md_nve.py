"""NVE MD with a trained FACE model: stability + energy conservation (literature gap P4/§6.1, §6.15).

usage: python tools/md_nve.py <checkpoint> <mol> <T_init K> <n_steps> <dt fs>
Reports total-energy drift (meV/atom/ps), fluctuation, and whether the molecule stays bonded.
"""
import sys, time, json
import numpy as np
from ase import Atoms, units
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
sys.path.insert(0, ".")
from face import safety
from face.calculator import FACECalculator
from face.data import load_rmd17

safety.install()
ckpt, mol = sys.argv[1], sys.argv[2]
T, nsteps, dt = float(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
Z, R, *_ = load_rmd17(mol)
at = Atoms(numbers=Z, positions=R[0])
at.calc = FACECalculator(ckpt)
rng = np.random.default_rng(0)
MaxwellBoltzmannDistribution(at, temperature_K=T, rng=rng); Stationary(at); ZeroRotation(at)
d0 = at.get_all_distances()
bonded = d0 < 1.6
dyn = VelocityVerlet(at, timestep=dt * units.fs)
E, Tk, t0, broken = [], [], time.time(), False
for k in range(nsteps):
    dyn.run(1)
    if k % 10 == 0:
        E.append(at.get_potential_energy() + at.get_kinetic_energy())
        Tk.append(at.get_temperature())
        if (at.get_all_distances()[bonded] > 2.5).any():
            broken = True
            break
E = np.array(E)
t_ps = np.arange(len(E)) * 10 * dt / 1000
slope = np.polyfit(t_ps, E, 1)[0] if len(E) > 2 else float("nan")
n = len(at)
res = {"ckpt": ckpt, "mol": mol, "T0": T, "steps": k + 1, "dt_fs": dt, "sim_ps": (k + 1) * dt / 1000,
       "bond_broken": broken, "mean_T": float(np.mean(Tk)),
       "drift_meV_per_atom_per_ps": 1e3 * slope / n,
       "E_tot_std_meV_per_atom": 1e3 * E.std() / n,
       "E_tot_maxdev_meV_per_atom": 1e3 * np.abs(E - E[0]).max() / n,
       "wall_s": time.time() - t0}
print(json.dumps(res, indent=1))
