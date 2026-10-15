import numpy as np
import torch

KCAL2EV = 0.0433641153


def load_rmd17(mol, root="data/rmd17", split=1, n_train=950, n_val=50):
    d = np.load(f"{root}/rmd17_{mol}.npz")
    Z = d["nuclear_charges"].astype(np.int64)
    R = d["coords"].astype(np.float64)
    E = d["energies"].astype(np.float64) * KCAL2EV
    Fo = d["forces"].astype(np.float64) * KCAL2EV
    tr = np.loadtxt(f"{root}/index_train_0{split}.csv", dtype=np.int64)
    te = np.loadtxt(f"{root}/index_test_0{split}.csv", dtype=np.int64)
    rng = np.random.default_rng(split)
    tr = rng.permutation(tr)
    idx = {"train": tr[:n_train], "val": tr[n_train:n_train + n_val], "test": te}
    return Z, R, E, Fo, idx


class MoleculeSet:
    """Fixed-composition molecule trajectory, stored on device; yields padded-dense graphs."""

    def __init__(self, Z, R, E, Fo, device, dtype=torch.float64):
        self.N = len(Z)
        self.Z = torch.as_tensor(Z, device=device)
        self.R = torch.as_tensor(R, device=device, dtype=dtype)
        self.E = torch.as_tensor(E, device=device, dtype=dtype)
        self.F = torch.as_tensor(Fo, device=device, dtype=dtype)
        n = self.N
        full = torch.arange(n, device=device)[None].expand(n, n)
        off = ~torch.eye(n, dtype=torch.bool, device=device)
        self.nbr0 = full[off].view(n, n - 1)

    def __len__(self):
        return len(self.E)

    def batch(self, idx):
        idx = torch.as_tensor(idx, device=self.R.device)
        B, n = len(idx), self.N
        pos = self.R[idx].reshape(B * n, 3).clone()
        off = (torch.arange(B, device=pos.device) * n)[:, None, None]
        nbr = (self.nbr0[None] + off).reshape(B * n, n - 1)
        g = {
            "pos": pos,
            "Z": self.Z.repeat(B),
            "nbr": nbr,
            "shift": torch.zeros(B * n, n - 1, 3, device=pos.device, dtype=pos.dtype),
            "mask": torch.ones(B * n, n - 1, dtype=torch.bool, device=pos.device),
            "batch": torch.arange(B, device=pos.device).repeat_interleave(n),
            "E": self.E[idx],
            "F": self.F[idx].reshape(B * n, 3),
        }
        return g
