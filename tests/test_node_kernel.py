import sys
import torch
sys.path.insert(0, ".")
from face.node_kernel import node_algebra, node_algebra_ref
from face.cartesian import expand


def check(N=50, C=24, eq=True):
    torch.manual_seed(1)
    shapes = [(3,), (3,), (3,), (6,), (6,), (6,), (10,), (10,)]   # symmetric storage
    xs = [torch.randn(N, C, *s, device="cuda", requires_grad=True) for s in shapes]
    a = torch.randn(4, C, device="cuda", requires_grad=True)
    b = torch.randn(4, C, device="cuda", requires_grad=True)
    gI, gV, gM = (torch.randn(N, C, 8, device="cuda"), torch.randn(N, C, 3, device="cuda"),
                  torch.randn(N, C, 3, 3, device="cuda"))
    o1 = node_algebra(*xs, a, b, eq)
    ranks = [1, 1, 1, 2, 2, 2, 3, 3]
    o2 = node_algebra_ref(*[expand(x, l) for x, l in zip(xs, ranks)], a, b)
    L = lambda o: (o[0] * gI).sum() + ((o[1] * gV).sum() + (o[2] * gM).sum() if eq else 0)
    g1 = torch.autograd.grad(L(o1), xs + ([a, b] if eq else []), allow_unused=True)
    g2 = torch.autograd.grad(L(o2), xs + ([a, b] if eq else []), allow_unused=True)
    rel = lambda u, v: ((u - v).abs().max() / v.abs().max().clamp_min(1e-12)).item()
    names = ["inv", "V", "Mf"][: 3 if eq else 1] + ["dP1", "dQ1", "dS1", "dP2", "dQ2", "dS2", "dP3", "dQ3", "da", "db"]
    pairs = list(zip(list(o1[: 3 if eq else 1]) + list(g1), list(o2[: 3 if eq else 1]) + list(g2)))
    errs = [rel(u, v) for u, v in pairs]
    bad = [(n, e) for n, e in zip(names, errs) if e > 1e-4]
    assert not bad, bad
    print(f"eq={eq}: fused node algebra == reference (max rel err {max(errs):.1e} over outputs and gradients)")


if __name__ == "__main__":
    check(eq=True)
    check(eq=False)
