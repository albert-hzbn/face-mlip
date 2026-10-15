import sys
import numpy as np
import torch
sys.path.insert(0, ".")
from ase.build import bulk
from face.model import FACE
from face.datasets import GraphSet

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)


def energy_forces(m, at):
    at.info.setdefault("energy", 0.0); at.info.setdefault("charge", 0.0)
    at.arrays["forces"] = np.zeros((len(at), 3))
    g = GraphSet([at], m.rc, device="cpu", dtype=torch.float64, rc_long=10.0).batch([0])
    o = m(g)
    return o["energy"].item(), o["forces"].detach().numpy()


def run(terms, pme=False):
    m = FACE([11, 17], C=16, terms=terms, transport=True, eqmsg=True, factor=True, B=4, K=8)
    for p in m.parameters():
        p.data += 0.1 * torch.randn_like(p)
    for t in m.terms.values():
        if hasattr(t, "use_pme"):
            t.use_pme = pme
    # exact Ewald: symmetries to 1e-8; PME: mesh discretisation breaks them at ~1e-4 relative (documented)
    tolE = 1e-4 * 10 if pme else 1e-8
    tolF = 1e-3 if pme else 1e-7
    at = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    at.positions += np.random.default_rng(0).normal(0, 0.15, at.positions.shape)
    E, Fo = energy_forces(m, at)
    # wrap / lattice translation invariance
    b = at.copy(); b.positions += 3.1 * at.cell[0] / 5.64 + 0.37; b.wrap()
    E2, F2 = energy_forces(m, b)
    assert abs(E2 - E) < tolE and np.abs(F2 - Fo).max() < tolF, (E, E2)
    # extensivity
    E3, F3 = energy_forces(m, at.repeat((2, 1, 1)))
    assert abs(E3 - 2 * E) < max(1e-7, tolE) * (1 + abs(E)), (E3, 2 * E)
    assert np.abs(F3[:len(at)] - Fo).max() < tolF
    # rotation of positions + cell
    Q, _ = np.linalg.qr(np.random.default_rng(2).normal(size=(3, 3)))
    r = at.copy(); r.set_cell(at.cell.array @ Q.T, scale_atoms=False); r.positions = at.positions @ Q.T
    E4, F4 = energy_forces(m, r)
    assert abs(E4 - E) < tolE and np.abs(F4 - Fo @ Q.T).max() < tolF
    # finite-difference forces (periodic, incl. Ewald)
    h = 1e-5
    for i, c in [(0, 0), (5, 2)]:
        p, mm = at.copy(), at.copy()
        p.positions[i, c] += h; mm.positions[i, c] -= h
        fd = -(energy_forces(m, p)[0] - energy_forces(m, mm)[0]) / (2 * h)
        assert abs(fd - Fo[i, c]) < 1e-6 * (1 + abs(fd)), (fd, Fo[i, c])
    print(f"periodic terms={terms} pme={pme}: OK  E={E:.5f}")


if __name__ == "__main__":
    run(())
    run(("coulomb",))
    run(("qeq",))
    run(("zbl", "qeq", "disp"))
    run(("coulomb",), pme=True)
    run(("zbl", "coulomb", "disp"), pme=True)
