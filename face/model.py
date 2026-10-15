"""FACE: Factorised Cartesian Atomic Cluster Expansion.

E = sum_i E0(Z_i) + E_local + sum_k E_term_k
  E_local : Cartesian moment-tensor network (padded-dense neighbours, GEMM pooling)
  terms   : pluggable physics operators (ZBL, latent-charge electrostatics, dispersion, ...)

Graph format (padded dense):
  pos   (N,3)   positions (requires_grad for forces)
  Z     (N,)    atomic numbers
  nbr   (N,M)   neighbour indices (padding -> any valid index, masked)
  shift (N,M,3) cartesian periodic shift vectors added to pos[nbr]-pos[i]
  mask  (N,M)   bool valid-neighbour mask
  batch (N,)    structure index
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cartesian import sym_monomials, expand, SLICES, SYM_MULT, const, Y2_WEIGHT
from .electrostatics import LatentCharges
from .dispersion import Dispersion
from .kernels import factor_density
from .node_kernel import node_algebra


# ----------------------------------------------------------------------------- basis
class BesselEnvelope(nn.Module):
    def __init__(self, n, rc, p=6):
        super().__init__()
        self.rc, self.p = rc, p
        self.register_buffer("freq", torch.arange(1, n + 1) * math.pi / rc)

    def envelope(self, r):
        p, x = self.p, r / self.rc
        e = 1 - (p + 1) * (p + 2) / 2 * x ** p + p * (p + 2) * x ** (p + 1) - p * (p + 1) / 2 * x ** (p + 2)
        return e * (x < 1)

    def forward(self, r):
        return torch.sin(self.freq * r[..., None]) / r[..., None] * math.sqrt(2 / self.rc), self.envelope(r)


class UnitLinear(nn.Module):
    """Linear layer with unit-variance stored weights and an explicit 1/sqrt(fan_in) forward factor
    (e3nn-style). Same function at init as nn.Linear's scale, but Adam's per-parameter step is then
    ~lr *relative* to the weight instead of ~lr*sqrt(fan_in) -- the cause of divergence at lr >= 5e-3."""

    def __init__(self, fan_in, fan_out, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(fan_out, fan_in))
        self.bias = nn.Parameter(torch.zeros(fan_out)) if bias else None
        self.alpha = fan_in ** -0.5

    def forward(self, x):
        return F.linear(x, self.weight * self.alpha, self.bias)


def unit_parameterise(module):
    """Replace every nn.Linear inside `module` by an equivalent-at-init UnitLinear (in place)."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            u = UnitLinear(child.in_features, child.out_features, child.bias is not None).to(child.weight)
            with torch.no_grad():
                u.weight.copy_(child.weight / u.alpha)      # identical function to the original init
                if child.bias is not None:
                    u.bias.copy_(child.bias)
            setattr(module, name, u)
        else:
            unit_parameterise(child)
    return module


def mlp(sizes, act=nn.SiLU, last_act=False):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), act()]
    return nn.Sequential(*(layers if last_act else layers[:-1]))


class Affine(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.w = nn.Parameter(torch.ones(n)); self.b = nn.Parameter(torch.zeros(n))

    def forward(self, x):
        return x * self.w + self.b


# ----------------------------------------------------------------------------- local block
class MomentBlock(nn.Module):
    """One interaction pass.

    moments  A^l_{i,c} = sum_j R_{l,c}(r_ij) h_{j,c} u_ij^{(x)l}         (l = 0..3)
    products channel-diagonal contractions of learned linear projections P,Q,S of the
             moments (current + carried-over moments of previous passes) -> 2..5-body invariants
    """
    N_INV = 9

    def __init__(self, C, n_basis, n_in_moment_sets, radial_hidden=64, n_elem=10, K_dir=16, norm="layer", transport=False,
                 eq_in=False, eq_out=False):
        super().__init__()
        self.C, self.n_elem, self.K = C, n_elem, K_dir
        self.eq_in, self.eq_out = eq_in, eq_out
        self.saturate, self.sat_scale = False, 4.0
        # body-order standardisation: per-family RMS measured on the first training batch (fixed buffer).
        # Without it 4-/5-body invariants are ~1e-6..1e-9 of the 2-body ones and invisible after LayerNorm.
        self.standardise = False
        self.momnorm, self.mn_eps = False, 1e-6
        # v6 'calib': data-calibrated fixed scales (no scale-invariant layer -> no free scale direction)
        self.calib, self.clip_c, self.calib_ln = False, 10.0, False
        self.eq_linear = False
        self.max_body = 5
        self.elem_proj = False
        self.pqs_gate = nn.Embedding(n_elem, 4 * 3 * C)
        nn.init.ones_(self.pqs_gate.weight)
        self._zidx = None
        self.use_fused_node = False     # set by FACE.fuse() for inference
        self._calibrating = False
        self.register_buffer("mom_scale", torch.ones(4))
        self.register_buffer("eq_scale", torch.ones(2))
        self._lognorms = None
        self.register_buffer("inv_scale", torch.ones(self.N_INV))
        self.register_buffer("calibrated", torch.tensor(False))
        # radial: element-pair conditioned first layer (still a pure function of (r, Zi, Zj)
        # -> compilable to per-pair splines for inference)
        self.rad_in = nn.Linear(n_basis, radial_hidden)
        self.pair = nn.Embedding(n_elem * n_elem, radial_hidden)
        nn.init.zeros_(self.pair.weight)
        self.directional = n_in_moment_sets > 1
        self.transport = transport and self.directional
        self.radial = mlp([radial_hidden, radial_hidden, (8 if self.transport else 4) * C])
        cin = C * n_in_moment_sets
        # projections per rank; 3 independent projections (P, Q, S)
        self.proj = nn.ModuleList([nn.Linear(cin, 3 * C, bias=False) for _ in range(4)])
        self.elem_scale = nn.Embedding(n_elem, 4 * C)
        nn.init.ones_(self.elem_scale.weight)
        self.out = mlp([self.N_INV * C + C + 4, 2 * C, 2 * C, C + 1])
        self._out_in = self.N_INV * C + C + 4
        if norm == "layer":
            self.norm = nn.LayerNorm(self.N_INV * C)
        elif norm == "affine":      # magnitude-preserving: learned per-invariant scale only
            self.norm = Affine(self.N_INV * C)
        else:
            self.norm = nn.Identity()
        # bond-projected directional invariants of previous-pass tensors (pass >= 2)
        if self.directional:
            cprev = C if eq_in else C * (n_in_moment_sets - 1)
            self.pv = nn.Linear(cprev, K_dir, bias=False)
            self.pm = nn.Linear(cprev, K_dir, bias=False)
            self.dir_mix = nn.Linear(4 * K_dir, C)
        if self.transport:      # rank-preserving tensor messages (Cartesian 1x1, 2x1 paths)
            self.tv = nn.Linear(cprev, C, bias=False)
            self.tm = nn.Linear(cprev, C, bias=False)
        if eq_out:              # gated equivariant products -> higher-body-order tensors for next pass
            self.vpath = nn.Parameter(torch.ones(4, C))
            self.mpath = nn.Parameter(torch.ones(4, C))
            self.gate = nn.Linear(self.N_INV * C, 2 * C)
            nn.init.zeros_(self.gate.weight); nn.init.ones_(self.gate.bias)

    def moments(self, h, rb, Y, nbr, env, zidx, avg_nb, prev_A=(), eq=None):
        N, M = nbr.shape
        flat = nbr.reshape(-1)
        gather = lambda x: x.index_select(0, flat).view(N, M, *x.shape[1:])
        zp = zidx[:, None] * self.n_elem + gather(zidx)
        x = F.silu(self.rad_in(rb) + self.pair(zp))
        R = self.radial(x).view(N, M, -1, self.C)                    # (N,M,4|8,C)
        hj = gather(h)
        if self.directional:
            src1 = eq[0] if self.eq_in else torch.cat([A[1] for A in prev_A], 1)
            src2 = eq[1] if self.eq_in else torch.cat([A[2] for A in prev_A], 1)
            V = self.pv(src1.transpose(1, 2)).transpose(1, 2)              # (N,K,3)
            Ms = self.pm(src2.transpose(1, 2)).transpose(1, 2)             # (N,K,6)
            u = Y[:, :, 1:4]
            Y2 = Y[:, :, 4:10] * const(Y2_WEIGHT, Y)
            d = torch.cat([
                torch.einsum("nma,nmka->nmk", u, gather(V)),            # u . V_j
                torch.einsum("nma,nmka->nmk", Y2, gather(Ms)),          # u^T M_j u
                torch.einsum("nma,nka->nmk", u, V),                  # u . V_i
                torch.einsum("nma,nka->nmk", Y2, Ms),                # u^T M_i u
            ], -1)
            hj = hj + self.dir_mix(d)
        w = R[:, :, :4] * hj[:, :, None, :] * env[..., None, None]    # env = 0 beyond r_c
        s = self.elem_scale(zidx).view(N, 4, self.C)
        # one GEMM over all ranks (N,4C,M)@(N,M,20); keep the rank-diagonal blocks
        Af = torch.bmm(w.reshape(N, M, 4 * self.C).transpose(1, 2), Y).view(N, 4, self.C, -1)
        Af = Af * (s[..., None] / avg_nb)
        A = [Af[:, l, :, SLICES[l]] for l in range(4)]
        if self.transport:
            Vt = gather(self.tv(src1.transpose(1, 2)).transpose(1, 2))         # (N,M,C,3)
            Mt = gather(self.tm(src2.transpose(1, 2)).transpose(1, 2))         # (N,M,C,6) sym
            Mf = expand(Mt, 2)                                                 # (N,M,C,3,3)
            u = Y[:, :, None, 1:4]                                             # (N,M,1,3)
            Rt = R[:, :, 4:] * env[..., None, None]                            # (N,M,4,C)
            Mu = (Mf * u[..., None, :]).sum(-1)                                # M_j u
            uV = u * Vt                                                        # diag of u (x) V
            sym_uV = torch.cat([uV, 0.5 * (u[..., [0, 0, 1]] * Vt[..., [1, 2, 2]]
                                           + u[..., [1, 2, 2]] * Vt[..., [0, 0, 1]])], -1)
            A1 = (Rt[:, :, 0, :, None] * Vt + Rt[:, :, 1, :, None] * Mu).sum(1)
            A2 = (Rt[:, :, 2, :, None] * Mt + Rt[:, :, 3, :, None] * sym_uV).sum(1)
            A[1] = A[1] + A1 * (s[:, 1, :, None] / avg_nb)
            A[2] = A[2] + A2 * (s[:, 2, :, None] / avg_nb)
        return A

    def invariants(self, A_sets):
        C = self.C
        # concatenate moment sets along channels, project with P,Q,S per rank
        Pq, lognorms = [], []
        for l in range(4):
            x = torch.cat([A[l] for A in A_sets], dim=1)                  # (N,cin,k_l)
            if self.calib:
                if self._calibrating:
                    with torch.no_grad():
                        rms = ((x * x * const(SYM_MULT[l], x)).sum(-1).mean()).sqrt().clamp_min(1e-12)
                        self.mom_scale[l] = rms
                x = x / self.mom_scale[l].clone()     # clone: buffer is written in-place during calibration
            if self.momnorm:
                # equivariant RMS norm over channels (Frobenius in sym storage): removes the free overall
                # scale that let products of order up to ~15 run away; the norm itself is kept as a feature
                n2 = (x * x * const(SYM_MULT[l], x)).sum(-1).mean(1, keepdim=True)        # (N,1)
                x = x * torch.rsqrt(n2 + self.mn_eps)[..., None]
                lognorms.append(torch.log1p(n2 / self.mn_eps))
            if self.saturate:
                # equivariant saturation: per-channel scalar factor 1/sqrt(1+|A|^2/s^2) bounds all
                # 2..5-body products (float32 overflow guard), smooth and rotation-invariant
                n2 = (x * x * const(SYM_MULT[l], x)).sum(-1, keepdim=True)   # Frobenius norm in sym storage
                x = x * torch.rsqrt(1 + n2 / self.sat_scale ** 2)
            y = self.proj[l](x.transpose(1, 2)).transpose(1, 2)           # (N,3C,k_l)
            if self.elem_proj:
                # centre-element-dependent product weights (MACE-like): per-element gate on P,Q,S channels
                y = y * self.pqs_gate(self._zidx).view(-1, 4, 3 * C)[:, l, :, None]
            if self.use_fused_node and not self.eq_linear:
                Pq.append(y.split(C, dim=1))                              # fused kernel takes sym storage
            else:
                Pq.append(expand(y, l).split(C, dim=1))                   # full tensor
        (P0, Q0, S0), (P1, Q1, S1), (P2, Q2, S2), (P3, Q3, S3) = Pq
        P0 = P0.squeeze(-1)
        if self.use_fused_node and not self.eq_linear:
            # fused CUDA node algebra (inference): all families + gated-product paths in registers
            N = P0.shape[0]
            zero = P1.new_zeros(4, C)
            a = self.vpath if self.eq_out else zero
            b = self.mpath if self.eq_out else zero
            inv8, Vf, Mff = node_algebra(P1, Q1, S1, P2, Q2, S2, P3, Q3, a, b, self.eq_out)
            fams = [P0] + list(inv8.unbind(-1))
            if self.max_body < 5:
                keep = [2, 3, 3, 3, 4, 4, 4, 4, 5]
                fams = [x if bo <= self.max_body else torch.zeros_like(x) for x, bo in zip(fams, keep)]
            self._lognorms = torch.cat(lognorms, 1) if lognorms else None
            return torch.cat(fams, dim=1), ("fused", Vf, Mff)
        # contractions written as broadcast multiply-sum over the 3-dims: torch lowers the
        # equivalent einsums to millions of tiny bmm's, ~10x slower (see tools/profile_face.py)
        PQ2 = (P2[..., :, :, None] * Q2[..., None, :, :]).sum(-2)
        X33 = (P3[..., :, :, :, None] * Q3[..., :, :, None, :]).sum((2, 3))
        inv = [
            P0,                                                         # 2-body
            (P1 * Q1).sum(-1),                                          # 3-body
            (P2 * Q2).sum((-1, -2)),
            (P3 * Q3).sum((-1, -2, -3)),
            ((P1[..., :, None] * Q2).sum(-2) * S1).sum(-1),             # 4-body  P1.Q2.S1
            (PQ2 * S2.transpose(-1, -2)).sum((-1, -2)),                 #         tr(P2 Q2 S2)
            ((P3 * Q2[..., None, :, :]).sum((-1, -2)) * S1).sum(-1),    #         P3:Q2 . S1
            (X33 * S2).sum((-1, -2)),                                   #         P3:Q3 : S2
            (((P3 * Q1[..., None, None, :]).sum(-1) * S1[..., None, :]).sum(-1) * Q1).sum(-1),  # 5-body
        ]
        self._lognorms = torch.cat(lognorms, 1) if lognorms else None                # (N,4)
        if self.max_body < 5:
            # family body orders: [2,3,3,3,4,4,4,4,5]; dropped families are zeroed (keeps tensor layout)
            keep = [2, 3, 3, 3, 4, 4, 4, 4, 5]
            inv = [x if bo <= self.max_body else torch.zeros_like(x) for x, bo in zip(inv, keep)]
        return torch.cat(inv, dim=1), (P1, Q1, S1, P2, Q2, S2, P3, PQ2)

    def equivariant_out(self, inv, T):
        if T[0] == "fused":
            Mf = 0.5 * (T[2] + T[2].transpose(-1, -2))
            Ms = torch.stack([Mf[..., 0, 0], Mf[..., 1, 1], Mf[..., 2, 2],
                              Mf[..., 0, 1], Mf[..., 0, 2], Mf[..., 1, 2]], -1)
            gv, gm = self.gate(inv).chunk(2, dim=-1)
            return self._eq_finish(T[1] * gv[..., None], Ms * gm[..., None])
        P1, Q1, S1, P2, Q2, S2, P3, PQ2 = T
        a, b = self.vpath[..., None], self.mpath[..., None, None]
        if self.eq_linear:
            # degree cap: only linear (degree-1) projections of the moments are carried to the next pass
            V, Mf = a[0] * P1, b[0] * P2
            Ms = torch.stack([Mf[..., 0, 0], Mf[..., 1, 1], Mf[..., 2, 2],
                              Mf[..., 0, 1], Mf[..., 0, 2], Mf[..., 1, 2]], -1)
            gv, gm = self.gate(inv).chunk(2, dim=-1)
            return self._eq_finish(V * gv[..., None], Ms * gm[..., None])
        V = (a[0] * P1 + a[1] * (Q2 * S1[..., None, :]).sum(-1)
             + a[2] * (P3 * Q2[..., None, :, :]).sum((-1, -2)) + a[3] * (P2 * Q1[..., None, :]).sum(-1))
        Mf = (b[0] * P2 + b[1] * P1[..., :, None] * Q1[..., None, :] + b[2] * PQ2
              + b[3] * (P3 * S1[..., None, None, :]).sum(-1))
        Mf = 0.5 * (Mf + Mf.transpose(-1, -2))
        Ms = torch.stack([Mf[..., 0, 0], Mf[..., 1, 1], Mf[..., 2, 2],
                          Mf[..., 0, 1], Mf[..., 0, 2], Mf[..., 1, 2]], -1)
        gv, gm = self.gate(inv).chunk(2, dim=-1)
        return self._eq_finish(V * gv[..., None], Ms * gm[..., None])

    def _eq_finish(self, V, Ms):
        if self.calib:
            if self._calibrating:
                with torch.no_grad():
                    self.eq_scale[0] = (V * V).sum(-1).mean().sqrt().clamp_min(1e-12)
                    self.eq_scale[1] = (Ms * Ms * const(SYM_MULT[2], Ms)).sum(-1).mean().sqrt().clamp_min(1e-12)
            V, Ms = V / self.eq_scale[0].clone(), Ms / self.eq_scale[1].clone()
        if self.momnorm:      # products are degree 2-3 in the moments: renormalise before transport
            V = V * torch.rsqrt((V * V).sum(-1).mean(1, keepdim=True) + self.mn_eps)[..., None]
            Ms = Ms * torch.rsqrt((Ms * Ms * const(SYM_MULT[2], Ms)).sum(-1).mean(1, keepdim=True) + self.mn_eps)[..., None]
        return V, Ms

    def forward(self, h, rb, Y, nbr, env, zidx, avg_nb, prev_A, eq=None):
        self._zidx = zidx
        self._calibrating = self.calib and self.training and not bool(self.calibrated)
        A = self.moments(h, rb, Y, nbr, env, zidx, avg_nb, prev_A, eq)
        inv, T = self.invariants(prev_A + [A])
        if self.calib:
            N = inv.shape[0]
            fam = inv.view(N, self.N_INV, -1)
            if self._calibrating:
                with torch.no_grad():
                    self.inv_scale.copy_(fam.pow(2).mean((0, 2)).sqrt().clamp_min(1e-12))
            fam = fam / self.inv_scale.clone()[None, :, None]
            inv = (fam * torch.rsqrt(1 + (fam / self.clip_c) ** 2)).view(N, -1)   # smooth overflow valve
        if self.standardise:
            N = inv.shape[0]
            fam = inv.view(N, self.N_INV, -1)
            if self.training and not bool(self.calibrated):
                with torch.no_grad():
                    self.inv_scale.copy_(fam.pow(2).mean((0, 2)).sqrt().clamp_min(1e-30))
                    self.calibrated.fill_(True)
            inv = (fam / self.inv_scale[None, :, None]).view(N, -1)
        if not self.calib or self.calib_ln:
            inv = self.norm(inv)
        ln = self._lognorms if self._lognorms is not None else inv.new_zeros(inv.shape[0], 4)
        o = self.out(torch.cat([inv, h, ln], dim=1))
        eq_new = self.equivariant_out(inv, T) if self.eq_out else None
        if self._calibrating:
            self.calibrated.fill_(True)
            self._calibrating = False
        return o[:, :1], h + o[:, 1:], A, eq_new


def _transport_map():
    """(36 = u-power*9 + component, 18 = [sum phi V (3) | sum phi M u (3) | sum phi M (6) | sum phi sym(u V) (6)])."""
    sym = {(0, 0): 0, (1, 1): 1, (2, 2): 2, (0, 1): 3, (1, 0): 3, (0, 2): 4, (2, 0): 4, (1, 2): 5, (2, 1): 5}
    pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    T = torch.zeros(36, 18, dtype=torch.float64)
    for al in range(3):
        T[al, al] = 1.0                                           # sum phi V_alpha
        for be in range(3):
            T[(be + 1) * 9 + 3 + sym[(al, be)], 3 + al] += 1.0    # sum phi M_{alpha beta} u_beta
    for k in range(6):
        T[3 + k, 6 + k] = 1.0                                     # sum phi M
        p, q = pairs[k]
        if p == q:
            T[(p + 1) * 9 + p, 12 + k] = 1.0                      # sum phi u_p V_p
        else:
            T[(p + 1) * 9 + q, 12 + k] += 0.5                     # 0.5 (u_p V_q + u_q V_p)
            T[(q + 1) * 9 + p, 12 + k] += 0.5
    return T


TRANSPORT_MAP = _transport_map()


def _transport_map3():
    """(90 = u-monomial(10: 1,u,uu-sym)*9 + component, 38 = [V(3) | Mu(3) | M(6) | sym uV(6) | sym uM(10) | sym uuV(10)])."""
    from .cartesian import SYM_IDX
    s2 = {(0, 0): 0, (1, 1): 1, (2, 2): 2, (0, 1): 3, (1, 0): 3, (0, 2): 4, (2, 0): 4, (1, 2): 5, (2, 1): 5}
    T = torch.zeros(90, 38, dtype=torch.float64)
    base = _transport_map()                       # (36,18) on monomials [1, u_x, u_y, u_z] -> p = 0..3
    T[:36, :18] = base
    for k3, (a, b, c) in enumerate(SYM_IDX[3]):
        for (x, y, z) in ((a, b, c), (b, a, c), (c, a, b)):     # sym(u (x) M): (u_a M_bc + u_b M_ac + u_c M_ab)/3
            T[(1 + x) * 9 + 3 + s2[(y, z)], 18 + k3] += 1.0 / 3
        for (x, y, z) in ((a, b, c), (a, c, b), (b, c, a)):     # sym(u u V): (u_a u_b V_c + u_a u_c V_b + u_b u_c V_a)/3
            T[(4 + s2[(x, y)]) * 9 + z, 28 + k3] += 1.0 / 3
    return T


TRANSPORT_MAP3 = _transport_map3()


class RadialSpline(nn.Module):
    """phi(r, pair) from a table: values T and exact derivatives dT on a uniform grid (cubic Hermite, C1)."""

    def __init__(self, T, dT, dr):
        super().__init__()
        self.register_buffer("T", T)        # (P, G, B)
        self.register_buffer("dT", dT)
        self.dr = dr

    @staticmethod
    def build(block, basis, rc, n_grid):
        dev = block.rad_in.weight.device
        dt = block.rad_in.weight.dtype
        r = torch.linspace(0, rc, n_grid, device=dev, dtype=dt).clamp_min(1e-4)
        P = block.pair.weight.shape[0]
        with torch.enable_grad():
            rr = r[None].expand(P, -1).clone().requires_grad_(True)
            bes, env = basis(rr)
            x = F.silu(block.rad_in(bes * env[..., None]) + block.pair.weight[:, None, :])
            T = block.radial(x)                                       # (P,G,B)
            dT = torch.stack([torch.autograd.grad(T[..., b].sum(), rr, retain_graph=True)[0]
                              for b in range(T.shape[-1])], -1)
        return RadialSpline(T.detach(), dT.detach(), rc / (n_grid - 1))   # not r[1]-r[0]: r[0] is clamped

    def forward(self, r, pair):
        P, G, B = self.T.shape
        x = (r / self.dr).clamp(0, G - 1)
        i = torch.floor(x).long().clamp(max=G - 2)     # integer clamp: G-1-1e-6 rounds to G-1 in float32
        t = (x - i)[..., None]
        flat = (pair * G + i).reshape(-1)
        T, dT = self.T.reshape(P * G, B), self.dT.reshape(P * G, B)
        sh = (*r.shape, B)
        p0, p1 = T.index_select(0, flat).view(sh), T.index_select(0, flat + 1).view(sh)
        m0, m1 = dT.index_select(0, flat).view(sh) * self.dr, dT.index_select(0, flat + 1).view(sh) * self.dr
        t2, t3 = t * t, t * t * t
        return (2 * t3 - 3 * t2 + 1) * p0 + (t3 - 2 * t2 + t) * m0 + (-2 * t3 + 3 * t2) * p1 + (t3 - t2) * m1


class FactorBlock(MomentBlock):
    """v4 interaction pass: edge work decoupled from node width C.

    edge weights   w_ij = phi_b(r_ij; Z_i, Z_j) (x) m_{j,k}      (B x K per edge, no C)
    density        D_i  = sum_j w_ij (x) u^{(x)l}                   one GEMM (N,BK,M)@(N,M,20)
    moments        A^l_i = W_l D^l_i                                 node-level channel map BK -> C
    transport      sum_j phi_b(r_ij) T_j  for T in {V_j, M_j u, M_j, sym(u (x) V_j)}  (GEMMs, K channels)
    The invariant / gated-product algebra is inherited unchanged from MomentBlock.
    """

    def __init__(self, C, n_basis, n_in_moment_sets, radial_hidden=64, n_elem=10, K_dir=16, norm="layer",
                 transport=False, eq_in=False, eq_out=False, B=16, K=32, cp=False):
        super().__init__(C, n_basis, n_in_moment_sets, radial_hidden, n_elem, K_dir, norm, transport,
                         eq_in, eq_out)
        self.cp = cp
        if cp:
            K = C          # v8 CP form: per-channel message, per-(rank,channel) radial
        self.B, self.Km = B, K
        self.use_fused = False          # set by FACE.fuse() for inference
        self.spline = None              # set by FACE.compile_radial() for inference
        self._r = None
        # v4: B shared radial functions; v8 (cp): B (transport) + 4C per-(rank, channel) radials
        self.radial = mlp([radial_hidden, radial_hidden, B + (4 * C if cp else 0)])
        self.msg = nn.Linear(C, K)
        self.mix = nn.ModuleList([nn.Linear(C if cp else B * K, C, bias=False) for _ in range(4)])
        if self.directional:
            self.dir_mix = nn.Linear(4 * K_dir, K)
            cprev = C if eq_in else C * (n_in_moment_sets - 1)
            nt = 2 if self.transport else 1
            del self.pv, self.pm
            self.pvt = nn.Linear(cprev, nt * K_dir, bias=False)    # [directional | transport]
            self.pmt = nn.Linear(cprev, nt * K_dir, bias=False)
        if self.transport:
            del self.tv, self.tm
            self.tmix1 = nn.Linear(B * 2 * K_dir, C, bias=False)
            self.tmix2 = nn.Linear(B * 2 * K_dir, C, bias=False)
            self.tmix3 = nn.Linear(B * 2 * K_dir, C, bias=False)   # used only with transport3
        self.transport3 = False

    def moments(self, h, rb, Y, nbr, env, zidx, avg_nb, prev_A=(), eq=None):
        N, M = nbr.shape
        B, K = self.B, self.Km
        flat = nbr.reshape(-1)
        gather = lambda x: x.index_select(0, flat).view(N, M, *x.shape[1:])
        zp = zidx[:, None] * self.n_elem + gather(zidx)
        if self.spline is not None:      # compiled radial: cubic Hermite table per element pair
            phi = self.spline(self._r, zp) * env[..., None]
        else:
            phi = self.radial(F.silu(self.rad_in(rb) + self.pair(zp))) * env[..., None]   # (N,M,B[+4C])
        if self.cp:
            Rcp = phi[..., B:].view(N, M, 4, self.C)
            phi = phi[..., :B]
        mj = gather(self.msg(h))                                                     # (N,M,K)
        if self.directional:
            src1 = eq[0] if self.eq_in else torch.cat([A[1] for A in prev_A], 1)
            src2 = eq[1] if self.eq_in else torch.cat([A[2] for A in prev_A], 1)
            Kd = self.K
            # one projection + one gather for both the directional (K_dir) and transport (K_dir) uses
            Vb = self.pvt(src1.transpose(1, 2)).transpose(1, 2)                      # (N,2Kd|Kd,3)
            Mb = self.pmt(src2.transpose(1, 2)).transpose(1, 2)                      # (N,2Kd|Kd,6)
            # one per-atom table [V(3) | M(6)] per channel, gathered once and split (backward = one concat)
            NT = torch.cat([Vb, Mb], -1)                                                  # (N,nt*Kd,9)
            G9 = gather(NT)                                                               # (N,M,nt*Kd,9)
            if self.transport:
                Gd, Gt = G9.split([Kd, Kd], dim=2)
            else:
                Gd, Gt = G9, None
            Vj, Mj = Gd.split([3, 6], dim=-1)
            V, Ms = Vb[:, :Kd], Mb[:, :Kd]
            u = Y[:, :, 1:4]
            Y2 = Y[:, :, 4:10] * const(Y2_WEIGHT, Y)
            d = torch.cat([(u[:, :, None] * Vj).sum(-1), (Y2[:, :, None] * Mj).sum(-1),
                           (u[:, :, None] * V[:, None]).sum(-1), (Y2[:, :, None] * Ms[:, None]).sum(-1)], -1)
            mj = mj + self.dir_mix(d)
        s = self.elem_scale(zidx).view(N, 4, self.C) / avg_nb
        if self.cp:
            # CP pooling: A_l[c,x] = sum_m R_{l,c}(r_m) mu_c(j_m) Y_x(u_m): edge cost 4C, one GEMM per rank
            w = (Rcp * mj[:, :, None, :]).reshape(N, M, 4 * self.C)                  # (N,M,4C) contiguous
            # one GEMM against all 20 monomials, keep the rank-diagonal blocks: slicing the big edge tensor per
            # rank cost a full copy (+ zero-fill in backward) per rank -- 32 % of CP inference
            Af = torch.bmm(Y.transpose(1, 2), w).view(N, 20, 4, self.C)               # (N,20,4,C)
            A = [self.mix[l](Af[:, SLICES[l], l]).transpose(1, 2) * s[:, l, :, None] for l in range(4)]
        elif self.use_fused:
            # fused CUDA kernel (inference): never materialises the per-edge B*K tensor
            Ds = factor_density(phi, mj, Y).split([1, 3, 6, 10], dim=1)              # (N,k_l,BK)
            A = [self.mix[l](Ds[l]).transpose(1, 2) * s[:, l, :, None] for l in range(4)]
        else:
            w = (phi[..., :, None] * mj[..., None, :]).reshape(N, M, B * K)
            Yt = Y.transpose(1, 2)                                                    # (N,20,M)
            # one GEMM per rank straight into (N,k_l,BK) -> channel map on the contiguous last dim;
            # slicing the small Y (not the big density) keeps the backward free of full-size zero fills
            A = [self.mix[l](torch.bmm(Yt[:, SLICES[l]], w)).transpose(1, 2) * s[:, l, :, None]
                 for l in range(4)]
        if self.transport:
            # all transport sums from ONE GEMM: left = [phi, phi*u_x, phi*u_y, phi*u_z] (N,4B,M),
            # right = gathered neighbour tensors [V_j | M_j] (N,M,Kd*9); M_j u and sym(u x V_j) are then
            # tiny node-level contractions -> no per-edge Mu / sym_uV / concat tensors
            # left (N,M,4B): [1,u_x,u_y,u_z] (x) phi ; right (N,M,Kd*9): per neighbour channel [V_j(3) | M_j(6)]
            nu = 10 if self.transport3 else 4          # bond monomials [1, u (, u(x)u)] = Y[..., :nu]
            L = (Y[:, :, :nu, None] * phi[..., None, :]).reshape(N, M, nu * B)
            Rt = Gt.reshape(N, M, Kd * 9)
            S = torch.bmm(L.transpose(1, 2), Rt) / avg_nb                                    # (N,nu*B,9Kd)
            # every transport moment is a fixed linear map of the (bond-monomial, component) sums:
            # rank<=2: (36,18) map; with transport3 also sym(u (x) M_j) and sym(u (x) u (x) V_j): (90,38) map
            X = S.view(N, nu, B, Kd, 9).permute(0, 2, 3, 1, 4).reshape(N, B, Kd, nu * 9)
            D = X @ const(TRANSPORT_MAP3 if self.transport3 else TRANSPORT_MAP, X)            # (N,B,Kd,18|38)
            D1 = D[..., :6].reshape(N, B, Kd, 2, 3).transpose(2, 3).reshape(N, B * 2 * Kd, 3)
            D2 = D[..., 6:18].reshape(N, B, Kd, 2, 6).transpose(2, 3).reshape(N, B * 2 * Kd, 6)
            if self.transport3:
                D3 = D[..., 18:38].reshape(N, B, Kd, 2, 10).transpose(2, 3).reshape(N, B * 2 * Kd, 10)
                A[3] = A[3] + self.tmix3(D3.transpose(1, 2)).transpose(1, 2) * (s[:, 3, :, None] * avg_nb)
            A[1] = A[1] + self.tmix1(D1.transpose(1, 2)).transpose(1, 2) * (s[:, 1, :, None] * avg_nb)
            A[2] = A[2] + self.tmix2(D2.transpose(1, 2)).transpose(1, 2) * (s[:, 2, :, None] * avg_nb)
        return A


# ----------------------------------------------------------------------------- physics terms
class ZBL(nn.Module):
    """Universal ZBL screened nuclear repulsion, switched off smoothly inside r_out."""
    def __init__(self, r_in=0.6, r_out=1.2):
        super().__init__()
        self.r_in, self.r_out = r_in, r_out

    def forward(self, g, r, feats):
        Zi = g["Z"].to(r.dtype)[:, None]
        Zj = g["Z"].to(r.dtype)[g["nbr"]]
        a = 0.46850 / (Zi ** 0.23 + Zj ** 0.23)
        x = r / a
        phi = (0.18175 * torch.exp(-3.19980 * x) + 0.50986 * torch.exp(-0.94229 * x)
               + 0.28022 * torch.exp(-0.40290 * x) + 0.02817 * torch.exp(-0.20162 * x))
        t = ((r - self.r_in) / (self.r_out - self.r_in)).clamp(0, 1)
        sw = 1 - t ** 3 * (10 - 15 * t + 6 * t ** 2)
        e = 14.399645 * Zi * Zj / r * phi * sw * g["mask"]
        return 0.5 * e.sum(1)


# ----------------------------------------------------------------------------- full model
class FACE(nn.Module):
    def __init__(self, elements, C=64, n_basis=8, rc=5.0, n_passes=2, terms=(),
                 radial_hidden=64, K_dir=16, norm="layer", transport=False, eqmsg=False,
                 factor=False, B=16, K=32, cp=False):
        super().__init__()
        self.rc, self.C = rc, C
        zmap = torch.full((119,), -1, dtype=torch.long)
        for k, z in enumerate(elements):
            zmap[z] = k
        self.register_buffer("zmap", zmap)
        n_el = len(elements)
        self.basis = BesselEnvelope(n_basis, rc)
        self.embed = nn.Embedding(n_el, C)
        Blk = FactorBlock if factor else MomentBlock
        kw = dict(B=B, K=K, cp=cp) if factor else {}
        self.blocks = nn.ModuleList([Blk(C, n_basis, t + 1, radial_hidden, n_el, K_dir, norm, transport,
                                         eq_in=eqmsg and t > 0, eq_out=eqmsg and t < n_passes - 1, **kw)
                                     for t in range(n_passes)])
        self.register_buffer("E0", torch.zeros(n_el))
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("avg_nb", torch.tensor(10.0))
        self.terms = nn.ModuleDict()
        for t in terms:
            if t == "zbl":
                self.terms[t] = ZBL()
            elif t == "coulomb":
                self.terms[t] = LatentCharges(C, mode="local")
            elif t == "qeq":
                self.terms[t] = LatentCharges(C, mode="full")
            elif t == "disp":
                self.terms[t] = Dispersion(C)
            else:
                raise ValueError(t)

    def set_transport3(self, on=True):
        for b in self.blocks:
            if hasattr(b, "transport3"):
                b.transport3 = on and b.transport
        return self

    def set_poly_readout(self, on=True):
        """Polynomial readout: linear readout in every pass except the last (energy and carried features stay
        polynomial in the atomic densities, ACE/MACE-like); the last pass keeps a small 16-unit MLP."""
        if not on:
            return self
        n = len(self.blocks)
        for k, b in enumerate(self.blocks):
            dev = next(b.parameters()).device
            b.out = (nn.Linear(b._out_in, b.C + 1) if k < n - 1 else mlp([b._out_in, 16, b.C + 1])).to(dev)
        return self

    def set_elem_proj(self, on=True):
        for b in self.blocks:
            b.elem_proj = on
        return self

    def set_max_body(self, n=5):
        for b in self.blocks:
            b.max_body = n
        return self

    def set_eq_linear(self, on=True):
        for b in self.blocks:
            b.eq_linear = on
        return self

    def set_calib(self, on=True, clip_c=10.0, keep_ln=False):
        for b in self.blocks:
            b.calib, b.clip_c, b.calib_ln = on, clip_c, keep_ln
        return self

    def set_momnorm(self, on=True, eps=1e-6):
        for b in self.blocks:
            b.momnorm, b.mn_eps = on, eps
        return self

    def set_standardise(self, on=True):
        for b in self.blocks:
            b.standardise = on
        return self

    def set_saturation(self, on=True, scale=4.0):
        for b in self.blocks:
            b.saturate, b.sat_scale = on, scale
        return self

    @torch.no_grad()
    def compile_radial(self, n_grid=4096, on=True):
        """Tabulate each factorised block's learned radial basis phi_b(r; Z_i, Z_j) (before the envelope)
        on a grid per element pair; inference then uses cubic Hermite interpolation (C1) instead of the MLP."""
        for b in self.blocks:
            if not hasattr(b, "spline"):
                continue
            if not on:
                b.spline = None
                continue
            b.spline = RadialSpline.build(b, self.basis, self.rc, n_grid)
        return self

    def fuse(self, on=True):
        """Switch factorised blocks to the fused CUDA kernels (first-order grads only: inference/MD)."""
        for b in self.blocks:
            if hasattr(b, "use_fused"):
                b.use_fused = on
            b.use_fused_node = on
        return self

    def forward(self, g, forces=True, create_graph=False):
        pos = g["pos"]
        if forces and not pos.requires_grad:
            pos.requires_grad_(True)
        nbr, mask = g["nbr"], g["mask"]
        vec = pos.index_select(0, nbr.reshape(-1)).view(*nbr.shape, 3) - pos[:, None] + g["shift"]
        r2 = (vec * vec).sum(-1)
        r = torch.sqrt(torch.where(mask, r2, torch.ones_like(r2)))
        u = vec / r[..., None]
        bes, env = self.basis(r)
        env = env * mask
        rb = bes * env[..., None]
        Y = sym_monomials(u)
        zidx = self.zmap[g["Z"]]
        h = self.embed(zidx)
        e_at = self.E0[zidx].clone()
        A_sets, eq = [], None
        for blk in self.blocks:
            blk._r = r
            e, h, A, eq = blk(h, rb, Y, nbr, env, zidx, self.avg_nb, A_sets, eq)
            A_sets.append(A)
            e_at = e_at + self.scale * e.squeeze(-1)
        for t in self.terms.values():
            e_at = e_at + t(g, r, h)
        nstr = g["nstr"] if "nstr" in g else int(g["batch"].max()) + 1   # pass nstr to stay sync-free (CUDA graphs)
        E = torch.zeros(nstr, device=pos.device, dtype=e_at.dtype).index_add(0, g["batch"], e_at)
        out = {"energy": E, "atom_energy": e_at}
        if forces:
            (grad,) = torch.autograd.grad(E.sum(), pos, create_graph=create_graph)
            out["forces"] = -grad
        return out
