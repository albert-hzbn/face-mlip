"""Latent-charge electrostatics operator (pluggable FACE term).

Atoms carry Gaussian charges (width sigma) predicted from the local features:
    E_es = sum_i chi_i q_i + 1/2 sum_i J_i q_i^2 + 1/2 k_e sum_ij q_i q_j A_ij ,   sum_i q_i = Q
A_ij = smeared Coulomb kernel erf(r/gamma)/r, gamma = 2 sigma (sum over periodic images for PBC,
via Ewald with real-space kernel [erf(r/gamma) - erf(r/gamma')]/r and reciprocal kernel
4 pi/k^2 exp(-gamma'^2 k^2 / 4)).

mode "local": q_i = -(chi_i - mu)/J_i  (global chemical potential mu; O(N) charges, Ewald energy
              O(N K + N M)) -- total-charge aware, non-local charge redistribution through mu.
mode "full" : q = argmin E_es  -> linear system [[J + k_e A, 1], [1^T, 0]] [q, -mu] = [-chi, Q]
              (4G-HDNNP-style charge equilibration with electrostatic coupling; dense O(N^3) here,
              CG + PME for large systems is a drop-in replacement of the matvec).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

KE = 14.399645  # eV * Angstrom / e^2


def _kvectors(cell, kcut):
    rec = 2 * math.pi * torch.linalg.inv(cell).T          # rows = reciprocal vectors
    nmax = [int(math.ceil(kcut / rec[i].norm().item())) for i in range(3)]
    rng = [torch.arange(-n, n + 1, device=cell.device, dtype=cell.dtype) for n in nmax]
    n = torch.stack(torch.meshgrid(*rng, indexing="ij"), -1).reshape(-1, 3)
    k = n @ rec
    k2 = (k * k).sum(-1)
    keep = (k2 > 1e-12) & (k2 < kcut ** 2)
    return k[keep], k2[keep]


def _bspline4(u):
    """Order-4 cardinal B-spline weights for the 4 grid points floor(u)-3..floor(u) (u: (...,))."""
    f = u - torch.floor(u)
    w0 = (1 - f) ** 3 / 6
    w1 = (3 * f ** 3 - 6 * f ** 2 + 4) / 6
    w2 = (-3 * f ** 3 + 3 * f ** 2 + 3 * f + 1) / 6
    w3 = f ** 3 / 6
    return torch.stack([w3, w2, w1, w0], -1)            # weights for offsets 0,-1,-2,-3


def _bmod2(n, device, dtype):
    """|b(m)|^2 Euler-spline modulus for order 4 (Essmann et al. 1995)."""
    M4 = torch.tensor([1 / 6, 4 / 6, 1 / 6], dtype=torch.float64)       # M_4(1), M_4(2), M_4(3)
    m = torch.arange(n, dtype=torch.float64)
    k = torch.arange(3, dtype=torch.float64)
    arg = 2 * math.pi * m[:, None] * k[None] / n
    den = (M4 * torch.cos(arg)).sum(-1) ** 2 + (M4 * torch.sin(arg)).sum(-1) ** 2
    return (1.0 / den.clamp_min(1e-10)).to(device=device, dtype=dtype)


def pme_recip_energy(q, pos, cell, gp, spacing=None):
    """Smooth-PME reciprocal energy 1/(2V) sum_k 4pi/k^2 exp(-gp^2 k^2/4) |S(k)|^2  (O(N + G log G))."""
    spacing = spacing or gp / 4.0          # h^4 convergence: gp/4 -> ~1e-4 relative (tests)
    lengths = cell.norm(dim=-1)
    n = [max(8, int(math.ceil(L.item() / spacing))) for L in lengths]
    inv = torch.linalg.inv(cell)
    frac = pos @ inv                                                     # (N,3) fractional
    u = frac * pos.new_tensor(n)
    base = torch.floor(u).long()
    W = [_bspline4(u[:, d]) for d in range(3)]                           # 3 x (N,4)
    off = torch.arange(0, -4, -1, device=pos.device)
    idx = [(base[:, d, None] + off) % n[d] for d in range(3)]            # 3 x (N,4)
    w = (q[:, None, None, None] * W[0][:, :, None, None] * W[1][:, None, :, None] * W[2][:, None, None, :])
    flat = (idx[0][:, :, None, None] * n[1] + idx[1][:, None, :, None]) * n[2] + idx[2][:, None, None, :]
    Q = torch.zeros(n[0] * n[1] * n[2], device=pos.device, dtype=pos.dtype).index_add(0, flat.reshape(-1), w.reshape(-1))
    F = torch.fft.fftn(Q.view(*n))
    ms = [torch.fft.fftfreq(n[d], d=1.0 / n[d], device=pos.device, dtype=pos.dtype) for d in range(3)]
    mg = torch.stack(torch.meshgrid(*ms, indexing="ij"), -1)            # integer m
    k = 2 * math.pi * mg @ inv.T                                         # k = 2 pi m . reciprocal
    k2 = (k * k).sum(-1)
    V = torch.det(cell).abs()
    B = _bmod2(n[0], pos.device, pos.dtype)[:, None, None] * _bmod2(n[1], pos.device, pos.dtype)[None, :, None]         * _bmod2(n[2], pos.device, pos.dtype)[None, None, :]
    G = torch.where(k2 > 1e-12, 4 * math.pi / k2.clamp_min(1e-12) * torch.exp(-gp ** 2 * k2 / 4), torch.zeros_like(k2))
    return 0.5 / V * (G * B * (F.real ** 2 + F.imag ** 2)).sum()


class LatentCharges(nn.Module):
    def __init__(self, C, sigma=0.5, mode="local", rc_real=5.0, acc=3.5):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(C, C), nn.SiLU(), nn.Linear(C, 2))
        self.sigma, self.mode = sigma, mode
        self.gamma = 2 * sigma
        self.gp = max(rc_real / acc, 1.05 * self.gamma)          # gamma' : Ewald splitting width
        self.kcut = 2 * acc / self.gp
        self.use_pme = True           # smooth PME for the reciprocal part ('coulomb' mode, PBC)

    # ---------------------------------------------------------------- kernels
    def _real(self, r):
        return (torch.erf(r / self.gamma) - torch.erf(r / self.gp)) / r

    def coulomb_matrix(self, pos, cell=None, nbr_pairs=None):
        """Dense A (n x n) for one structure."""
        n = len(pos)
        eye = torch.eye(n, device=pos.device, dtype=pos.dtype)
        if cell is None:
            # sqrt(d^2 + I), not norm(.) + I: the norm of the zero diagonal has a NaN second derivative
            d = torch.sqrt(((pos[:, None] - pos[None]) ** 2).sum(-1) + eye)
            return torch.erf(d / self.gamma) / d * (1 - eye)
        V = torch.det(cell).abs()
        k, k2 = _kvectors(cell, self.kcut)
        w = 4 * math.pi / V * torch.exp(-self.gp ** 2 * k2 / 4) / k2          # (K,)
        ph = pos @ k.T                                                         # (n,K)
        c, s = torch.cos(ph), torch.sin(ph)
        A = (c * w) @ c.T + (s * w) @ s.T
        i, j, sh = nbr_pairs
        r = (pos[j] - pos[i] + sh).norm(dim=-1)
        A = A.index_put((i, j), self._real(r), accumulate=True)
        A = A - eye * (2 / (math.sqrt(math.pi) * self.gp))                     # k-space self term
        return A - math.pi * self.gp ** 2 / V                                  # charged-cell background

    def ewald_energy(self, q, pos, cell, nbr_pairs):
        """0.5 q^T A q without forming A: O(N K + N M)."""
        V = torch.det(cell).abs()
        if self.use_pme:
            e = pme_recip_energy(q, pos, cell, self.gp)
        else:
            k, k2 = _kvectors(cell, self.kcut)
            w = 4 * math.pi / V * torch.exp(-self.gp ** 2 * k2 / 4) / k2
            ph = pos @ k.T
            Sc, Ss = q @ torch.cos(ph), q @ torch.sin(ph)
            e = 0.5 * (w * (Sc ** 2 + Ss ** 2)).sum()
        i, j, sh = nbr_pairs
        r = (pos[j] - pos[i] + sh).norm(dim=-1)
        e = e + 0.5 * (q[i] * q[j] * self._real(r)).sum()
        e = e - (q * q).sum() / (math.sqrt(math.pi) * self.gp)
        return e - 0.5 * math.pi * self.gp ** 2 / V * q.sum() ** 2

    # ---------------------------------------------------------------- forward
    def forward(self, g, r, feats):
        chi, hard = self.head(feats).unbind(-1)
        J = F.softplus(hard) + 0.1
        b, pos = g["batch"], g["pos"]
        nstr = int(b.max()) + 1
        Q = g.get("charge", None)
        e_atoms = torch.zeros_like(chi)
        qs = torch.zeros_like(chi)
        nbr, mask, shift = g["nbr"], g["mask"], g["shift"]
        periodic = "cell" in g and bool((g["cell"].abs().sum((1, 2)) > 0).any())
        if not periodic:
            return self._molecular_batch(chi, J, pos, b, nstr, Q, g)
        for s in range(nstr):
            idx = (b == s).nonzero().squeeze(1)
            p, x, Js = pos[idx], chi[idx], J[idx]
            Qs = Q[s] if Q is not None else p.new_zeros(())
            cell = g["cell"][s] if "cell" in g and g["cell"][s].abs().sum() > 0 else None
            pairs = None
            if cell is not None:
                off = idx[0]
                m = mask[idx]
                ii = torch.arange(len(idx), device=p.device)[:, None].expand_as(m)[m]
                pairs = (ii, nbr[idx][m] - off, shift[idx][m])
            if self.mode == "full":
                A = self.coulomb_matrix(p, cell, pairs)
                n = len(idx)
                H = torch.zeros(n + 1, n + 1, device=p.device, dtype=p.dtype)
                H[:n, :n] = KE * A + torch.diag(Js)
                H[:n, n] = 1; H[n, :n] = 1
                rhs = torch.cat([-x, Qs.reshape(1)])
                q = torch.linalg.solve(H, rhs)[:n]
                e_es = 0.5 * KE * q @ (A @ q)
            else:
                invJ = 1 / Js
                mu = (Qs + (x * invJ).sum()) / invJ.sum()
                q = -(x - mu) * invJ
                if cell is None:
                    e_es = 0.5 * KE * q @ (self.coulomb_matrix(p) @ q)
                else:
                    e_es = KE * self.ewald_energy(q, p, cell, pairs)
            # distribute the pair energy evenly over atoms of the structure (only totals are physical)
            e_atoms = e_atoms.index_add(0, idx, x * q + 0.5 * Js * q * q + e_es / len(idx))
            qs = qs.index_add(0, idx, q)
        g["q"] = qs
        return e_atoms

    def _molecular_batch(self, chi, J, pos, b, nstr, Q, g):
        """All non-periodic structures of a batch at once: block-diagonal Coulomb matrix."""
        N = len(chi)
        Qs = Q if Q is not None else chi.new_zeros(nstr)
        A = self.coulomb_matrix(pos) * (b[:, None] == b[None, :])
        if self.mode == "full":
            P = torch.zeros(nstr, N, device=chi.device, dtype=chi.dtype)
            P[b, torch.arange(N, device=chi.device)] = 1.0                       # structure membership
            H = torch.zeros(N + nstr, N + nstr, device=chi.device, dtype=chi.dtype)
            H[:N, :N] = KE * A + torch.diag(J)
            H[:N, N:] = P.T; H[N:, :N] = P
            q = torch.linalg.solve(H, torch.cat([-chi, Qs]))[:N]
        else:
            invJ = 1 / J
            num = torch.zeros(nstr, device=chi.device, dtype=chi.dtype).index_add(0, b, chi * invJ)
            den = torch.zeros(nstr, device=chi.device, dtype=chi.dtype).index_add(0, b, invJ)
            mu = (Qs + num) / den
            q = -(chi - mu[b]) * invJ
        g["q"] = q
        # per-atom split of the pair energy: 1/2 q_i sum_j A_ij q_j  (sums to the exact total)
        return chi * q + 0.5 * J * q * q + 0.5 * KE * q * (A @ q)
