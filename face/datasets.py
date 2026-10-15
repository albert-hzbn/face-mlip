"""Generic structure datasets -> padded-dense FACE batches."""
import numpy as np
import torch
from ase import Atoms
from ase.neighborlist import neighbor_list

BOHR = 0.529177210903
HARTREE = 27.211386245988


def read_runner(path):
    """RuNNer input.data (atomic units) -> list of ase.Atoms with info['energy'] (eV),
    info['charge'], arrays['forces'] (eV/A), arrays['ref_q'] (reference atomic charges)."""
    out, cur = [], None
    with open(path) as f:
        for line in f:
            t = line.split()
            if not t:
                continue
            k = t[0]
            if k == "begin":
                cur = {"lat": [], "pos": [], "sym": [], "q": [], "F": [], "E": 0.0, "Q": 0.0}
            elif k == "lattice":
                cur["lat"].append([float(x) for x in t[1:4]])
            elif k == "atom":
                cur["pos"].append([float(x) for x in t[1:4]]); cur["sym"].append(t[4])
                cur["q"].append(float(t[5])); cur["F"].append([float(x) for x in t[7:10]])
            elif k == "energy":
                cur["E"] = float(t[1])
            elif k == "charge":
                cur["Q"] = float(t[1])
            elif k == "end":
                pbc = len(cur["lat"]) == 3
                at = Atoms(symbols=cur["sym"], positions=np.array(cur["pos"]) * BOHR,
                           cell=np.array(cur["lat"]) * BOHR if pbc else None, pbc=pbc)
                at.info["energy"] = cur["E"] * HARTREE
                at.info["charge"] = cur["Q"]
                at.arrays["forces"] = np.array(cur["F"]) * HARTREE / BOHR
                at.arrays["ref_q"] = np.array(cur["q"])
                out.append(at)
    return out


class GraphSet:
    """Precomputes per-structure neighbour lists (CPU) and collates padded batches on device."""

    def __init__(self, atoms, rc, device="cuda", dtype=torch.float32, E0=None, rc_long=None):
        self.device, self.dtype, self.rc, self.rc_long = device, dtype, rc, rc_long
        self.items, self.long = [], []
        for at in atoms:
            i, j, S = neighbor_list("ijS", at, rc)
            sv = S @ at.cell.array if at.pbc.any() else np.zeros((len(i), 3))
            self.items.append((at, i, j, sv))
            if rc_long:
                li, lj, lS = neighbor_list("ijS", at, rc_long)
                self.long.append((li, lj, lS @ at.cell.array if at.pbc.any() else np.zeros((len(li), 3))))
        self.E0 = E0

    def __len__(self):
        return len(self.items)

    def batch(self, idx):
        rows, off = [], 0
        for b in idx:
            at, i, j, sv = self.items[b]
            cnt = np.bincount(i, minlength=len(at))
            rows.append((at, i, j, sv, cnt, off)); off += len(at)
        M = max(1, max(int(r[4].max()) if len(r[4]) else 1 for r in rows))
        N = off
        NB = np.zeros((N, M), np.int64); SH = np.zeros((N, M, 3)); MK = np.zeros((N, M), bool)
        pos, Z, bt, F, E, Q, cell = [], [], [], [], [], [], []
        for k, (at, i, j, sv, cnt, o) in enumerate(rows):
            n = len(at)
            if len(i):
                order = np.argsort(i, kind="stable"); i, j, sv = i[order], j[order], sv[order]
                start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
                slot = np.arange(len(i)) - start[i]
                NB[o + i, slot] = o + j; SH[o + i, slot] = sv; MK[o + i, slot] = True
            NB[o:o + n][~MK[o:o + n]] = o
            pos.append(at.positions); Z.append(at.numbers); bt.append(np.full(n, k))
            F.append(at.arrays["forces"])
            e = at.info["energy"] - (sum(self.E0[z] for z in at.numbers) if self.E0 else 0.0)
            E.append(e); Q.append(at.info.get("charge", 0.0))
            cell.append(at.cell.array if at.pbc.any() else np.zeros((3, 3)))
        t = lambda x, dt=self.dtype: torch.as_tensor(np.asarray(x), device=self.device, dtype=dt)
        extra = {}
        if self.rc_long:
            offs = [r[5] for r in rows]
            L = [self.long[b] for b in idx]
            extra = {"lr_i": t(np.concatenate([l[0] + o for l, o in zip(L, offs)]), torch.long),
                     "lr_j": t(np.concatenate([l[1] + o for l, o in zip(L, offs)]), torch.long),
                     "lr_shift": t(np.concatenate([l[2] for l in L]))}
        return {**extra,"pos": t(np.concatenate(pos)), "Z": t(np.concatenate(Z), torch.long),
                "nbr": t(NB, torch.long), "shift": t(SH), "mask": t(MK, torch.bool),
                "batch": t(np.concatenate(bt), torch.long), "F": t(np.concatenate(F)),
                "E": t(E), "charge": t(Q), "cell": t(np.stack(cell)),
                "natoms": t([len(r[0]) for r in rows])}


def fit_E0(atoms):
    """Per-element reference energies by least squares (float64)."""
    els = sorted({z for at in atoms for z in at.numbers})
    A = np.array([[np.sum(at.numbers == z) for z in els] for at in atoms], float)
    y = np.array([at.info["energy"] for at in atoms])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return {z: float(v) for z, v in zip(els, c)}
