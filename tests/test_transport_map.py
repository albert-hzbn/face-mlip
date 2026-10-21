"""Explicit per-edge check of the constant transport maps (rank <= 3)."""
import sys
import torch
sys.path.insert(0, ".")
from face.cartesian import sym_monomials, expand, SYM_IDX
from face.model import TRANSPORT_MAP3

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
E = 7
u = torch.nn.functional.normalize(torch.randn(E, 3), dim=-1)
V = torch.randn(E, 3)
Ms = torch.randn(E, 6)
M = expand(Ms, 2)
Y = sym_monomials(u)[:, :10]                                   # [1, u, uu]
X = (Y[:, :, None] * torch.cat([V, Ms], -1)[:, None, :]).reshape(E, 90)
D = X @ TRANSPORT_MAP3.to(X)
sym = lambda T3: torch.stack([T3[:, a, b, c] for (a, b, c) in SYM_IDX[3]], -1)
uM = torch.einsum("ea,ebc->eabc", u, M); uM = (uM + uM.permute(0, 2, 1, 3) + uM.permute(0, 3, 2, 1)) / 3
uuV = torch.einsum("ea,eb,ec->eabc", u, u, V); uuV = (uuV + uuV.permute(0, 1, 3, 2) + uuV.permute(0, 3, 2, 1)) / 3
Mu = torch.einsum("eab,eb->ea", M, u)
uV = torch.einsum("ea,eb->eab", u, V); uV = 0.5 * (uV + uV.transpose(1, 2))
sym2 = lambda T2: torch.stack([T2[:, 0, 0], T2[:, 1, 1], T2[:, 2, 2], T2[:, 0, 1], T2[:, 0, 2], T2[:, 1, 2]], -1)
ref = torch.cat([V, Mu, Ms, sym2(uV), sym(uM), sym(uuV)], -1)
err = (D - ref).abs().max().item()
assert err < 1e-12, err
# fully symmetric check: the symmetrised rank-3 tensors expand back consistently
assert torch.allclose(expand(sym(uM), 3), uM) and torch.allclose(expand(sym(uuV), 3), uuV)
print(f"transport map (rank<=3) == explicit per-edge tensors: max err {err:.1e}")
