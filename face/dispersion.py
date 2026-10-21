"""Environment-dependent dispersion operator (pluggable FACE term).

E_disp = -1/2 sum_{i != j, r < r_long} C6_ij / (r^6 + R0_ij^6) * s(r)
  C6_i  = c6_free(Z_i) * exp(w . h_i)       (learned modulation of free-atom C6 by local environment)
  C6_ij = sqrt(C6_i C6_j)                   (geometric combination rule, as in D3/TS)
  R0_ij = a1 * (R_i + R_j) / 2 + a2          (Becke-Johnson-type rational damping, learned a1, a2)
  s(r)  smooth switch to zero between r_long - 2 A and r_long
Uses a dedicated long-range pair list g["lr_i"], g["lr_j"], g["lr_shift"] (see datasets.GraphSet).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# free-atom C6 (Hartree*Bohr^6) and vdW radii (Bohr), Tkatchenko-Scheffler 2009 tables; unknown
# elements fall back to carbon -- the learned environment modulation absorbs the difference
_HA_B6 = 27.211386 * 0.529177 ** 6
_C6_AU = {1: 6.50, 6: 46.6, 7: 24.2, 8: 15.6, 11: 1556.0, 12: 627.0, 13: 528.0, 17: 94.6, 47: 339.0, 79: 298.0}
_R0 = {1: 3.10, 6: 3.59, 7: 3.34, 8: 3.19, 11: 3.74, 12: 3.66, 13: 3.84, 17: 3.71, 47: 3.82, 79: 3.86}  # Bohr->A below


class Dispersion(nn.Module):
    def __init__(self, C, r_long=10.0):
        super().__init__()
        self.r_long = r_long
        self.w = nn.Linear(C, 1)
        nn.init.zeros_(self.w.weight); nn.init.zeros_(self.w.bias)
        self.a1 = nn.Parameter(torch.tensor(0.4)); self.a2 = nn.Parameter(torch.tensor(2.5))
        c6 = torch.full((119,), _C6_AU[6] * _HA_B6)
        r0 = torch.full((119,), _R0[6] * 0.529177)
        for z, v in _C6_AU.items():
            c6[z] = v * _HA_B6
        for z, v in _R0.items():
            r0[z] = v * 0.529177
        self.register_buffer("c6", c6); self.register_buffer("r0", r0)

    def forward(self, g, r, feats):
        if "lr_i" not in g:
            raise KeyError("dispersion term needs a long-range pair list (GraphSet(rc_long=...))")
        Z, pos = g["Z"], g["pos"]
        i, j = g["lr_i"], g["lr_j"]
        d = (pos[j] - pos[i] + g["lr_shift"]).norm(dim=-1)
        C6i = self.c6[Z] * torch.exp(self.w(feats).squeeze(-1))
        C6 = torch.sqrt(C6i[i] * C6i[j])
        R0 = F.softplus(self.a1) * 0.5 * (self.r0[Z[i]] + self.r0[Z[j]]) + F.softplus(self.a2)
        t = ((d - (self.r_long - 2.0)) / 2.0).clamp(0, 1)
        sw = 1 - t ** 3 * (10 - 15 * t + 6 * t ** 2)
        e = -0.5 * C6 / (d ** 6 + R0 ** 6) * sw
        return torch.zeros_like(C6i).index_add(0, i, e)
