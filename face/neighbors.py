"""General (periodic) neighbour lists in the padded-dense layout used by FACE."""
import numpy as np
import torch
from ase.neighborlist import neighbor_list


def padded_graph(atoms_list, rc, device="cuda", dtype=torch.float32, skin=0.0):
    pos, Z, nbr, shift, mask, batch, off = [], [], [], [], [], [], 0
    rows = []
    for b, at in enumerate(atoms_list):
        n = len(at)
        i, j, S = neighbor_list("ijS", at, rc + skin)
        sv = S @ at.cell.array
        order = np.argsort(i, kind="stable")
        i, j, sv = i[order], j[order], sv[order]
        cnt = np.bincount(i, minlength=n)
        rows.append((n, i, j + off, sv, cnt))
        pos.append(at.positions); Z.append(at.numbers); batch.append(np.full(n, b))
        off += n
    M = max(int(r[4].max()) if len(r[4]) else 1 for r in rows)
    Ntot = off
    NB = np.zeros((Ntot, M), dtype=np.int64)
    SH = np.zeros((Ntot, M, 3))
    MK = np.zeros((Ntot, M), dtype=bool)
    base = 0
    for n, i, j, sv, cnt in rows:
        start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        slot = np.arange(len(i)) - start[i]
        NB[base + i, slot] = j
        SH[base + i, slot] = sv
        MK[base + i, slot] = True
        NB[base:base + n][~MK[base:base + n]] = base  # padding points at a valid atom
        base += n
    t = lambda x, dt=dtype: torch.as_tensor(x, device=device, dtype=dt)
    return {"pos": t(np.concatenate(pos)), "Z": t(np.concatenate(Z), torch.long),
            "nbr": t(NB, torch.long), "shift": t(SH), "mask": t(MK, torch.bool),
            "batch": t(np.concatenate(batch), torch.long)}
