import sys, math
import numpy as np
import torch
sys.path.insert(0, ".")
from ase.build import bulk
from ase.neighborlist import neighbor_list
from face.electrostatics import LatentCharges, KE

torch.set_default_dtype(torch.float64)


def pairs(at, rc):
    i, j, S = neighbor_list("ijS", at, rc)
    return (torch.tensor(i), torch.tensor(j), torch.tensor(S @ at.cell.array))


def test_madelung():
    at = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    q = torch.tensor([1.0 if s == "Na" else -1.0 for s in at.get_chemical_symbols()])
    pos, cell = torch.tensor(at.positions), torch.tensor(at.cell.array)
    ref = -1.747565 * KE / (5.64 / 2) * 4                     # 4 ion pairs in the cubic cell
    for rc in (5.0, 7.0):                                       # different Ewald splits
        es = LatentCharges(8, sigma=0.02, rc_real=rc); es.use_pme = False
        pr = pairs(at, rc)
        e1 = KE * es.ewald_energy(q, pos, cell, pr)
        e2 = 0.5 * KE * q @ (es.coulomb_matrix(pos, cell, pr) @ q)
        assert abs(e1 - e2) < 1e-8, (e1, e2)
        assert abs(e1 - ref) / abs(ref) < 1e-5, (e1.item(), ref)
        print(f"Madelung rc_real={rc}: E={e1.item():.6f} eV  ref={ref:.6f}  dense-vs-ewald {abs(e1-e2).item():.1e}")


def test_supercell_extensive():
    at = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    at.positions += np.random.default_rng(0).normal(0, 0.1, at.positions.shape)
    q = torch.tensor(np.random.default_rng(1).normal(0, 0.5, len(at)))
    q = q - q.mean()
    es = LatentCharges(8, sigma=0.5); es.use_pme = False
    e1 = es.ewald_energy(q, torch.tensor(at.positions), torch.tensor(at.cell.array), pairs(at, 5.0))
    sc = at.repeat((2, 1, 1))
    e2 = es.ewald_energy(q.repeat(2), torch.tensor(sc.positions), torch.tensor(sc.cell.array), pairs(sc, 5.0))
    assert abs(e2 - 2 * e1) < 1e-7 * abs(e1) + 1e-9, (e1, e2)
    print(f"supercell extensivity: E(2x)/E(1x) = {(e2 / e1).item():.8f}")


def test_pme():
    from ase import Atoms
    rng = np.random.default_rng(3)
    for L, n in [(12.0, 60), (20.0, 300)]:
        at = Atoms("X" * n, positions=rng.uniform(0, L, (n, 3)), cell=np.diag([L, 0.9 * L, 1.1 * L]), pbc=True)
        q = torch.tensor(rng.normal(0, 0.5, n)); q = q - q.mean()
        pos = torch.tensor(at.positions, requires_grad=True); cell = torch.tensor(at.cell.array)
        es = LatentCharges(8, sigma=0.5); es.use_pme = False
        pr = pairs(at, 5.0)
        e_ref = es.ewald_energy(q, pos, cell, pr); f_ref = -torch.autograd.grad(e_ref, pos)[0]
        es.use_pme = True
        e_pme = es.ewald_energy(q, pos, cell, pr); f_pme = -torch.autograd.grad(e_pme, pos)[0]
        de = abs((e_pme - e_ref) / e_ref).item(); df = ((f_pme - f_ref).norm() / f_ref.norm()).item()
        assert de < 1e-3 and df < 1e-3, (de, df)
        print(f"PME vs exact Ewald (L={L}, N={n}): rel dE {de:.1e}, rel dF {df:.1e}")


if __name__ == "__main__":
    test_pme()
    test_madelung()
    test_supercell_extensive()
