"""Cartesian moment-tensor algebra (no Clebsch-Gordan coefficients).

Edge directions u are lifted to symmetric outer powers u^{(x)l}, l = 0..3, stored in
*symmetric* (compressed) form: 1 + 3 + 6 + 10 = 20 components per edge instead of
1 + 3 + 9 + 27 = 40.  Node-level moments are expanded to full tensors only once per atom,
where all body-order contractions are evaluated with plain einsums.
"""
import itertools
import torch

SYM_IDX = {
    0: [()],
    1: [(0,), (1,), (2,)],
    2: [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)],
    3: [(0, 0, 0), (1, 1, 1), (2, 2, 2), (0, 0, 1), (0, 0, 2), (0, 1, 1),
        (1, 1, 2), (0, 2, 2), (1, 2, 2), (0, 1, 2)],
}
SLICES = {0: slice(0, 1), 1: slice(1, 4), 2: slice(4, 10), 3: slice(10, 20)}
NSYM = 20


def _expand_index(l):
    """Index map full tensor -> symmetric component for rank l."""
    lut = {t: k for k, t in enumerate(SYM_IDX[l])}
    full = []
    for t in itertools.product(range(3), repeat=l):
        full.append(lut[tuple(sorted(t))])
    return torch.tensor(full, dtype=torch.long)


EXPAND = {l: _expand_index(l) for l in range(4)}


def sym_monomials(u, lmax=3):
    """u: (..., 3) unit vectors -> (..., n_sym) symmetric monomials up to rank lmax."""
    x, y, z = u[..., 0], u[..., 1], u[..., 2]
    one = torch.ones_like(x)
    out = [one]
    if lmax >= 1:
        out += [x, y, z]
    if lmax >= 2:
        out += [x * x, y * y, z * z, x * y, x * z, y * z]
    if lmax >= 3:
        out += [x * x * x, y * y * y, z * z * z, x * x * y, x * x * z, x * y * y,
                y * y * z, x * z * z, y * z * z, x * y * z]
    return torch.stack(out, dim=-1)


# multiplicity of each symmetric component in the full tensor (Frobenius norm / full contractions)
SYM_MULT = {l: torch.bincount(EXPAND[l], minlength=len(SYM_IDX[l])).float() for l in range(4)}

EXPAND_MAT = {}
for _l in range(4):
    _m = torch.zeros(len(SYM_IDX[_l]), 3 ** _l)
    _m[EXPAND[_l], torch.arange(3 ** _l)] = 1.0
    EXPAND_MAT[_l] = _m


def expand(sym, l):
    """(..., k_l) symmetric components -> (..., 3, ..., 3) full rank-l tensor.
    Implemented as a GEMM with a constant 0/1 matrix: a gather along the innermost dimension
    (index_select) was the single most expensive kernel in the profile."""
    if l == 0:
        return sym
    E = const(EXPAND_MAT[l], sym)
    return (sym @ E).reshape(*sym.shape[:-1], *([3] * l))


_CONST = {}


def const(t, like):
    """Device/dtype-cached copy of a constant tensor (no host->device copy per call: CUDA-graph safe)."""
    key = (id(t), like.device, like.dtype)
    v = _CONST.get(key)
    if v is None:
        v = _CONST[key] = t.to(device=like.device, dtype=like.dtype)
    return v


Y2_WEIGHT = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])   # u^T M u from symmetric components
