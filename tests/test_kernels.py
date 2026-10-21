import sys, time
import torch
sys.path.insert(0, ".")
from face.kernels import factor_density, factor_density_ref


def check(N=300, M=37, B=16, K=32):
    torch.manual_seed(0)
    phi = torch.randn(N, M, B, device="cuda", requires_grad=True)
    msg = torch.randn(N, M, K, device="cuda", requires_grad=True)
    Y = torch.randn(N, M, 20, device="cuda", requires_grad=True)
    G = torch.randn(N, 20, B * K, device="cuda")
    D1 = factor_density(phi, msg, Y)
    g1 = torch.autograd.grad((D1 * G).sum(), (phi, msg, Y))
    D2 = factor_density_ref(phi, msg, Y)
    g2 = torch.autograd.grad((D2 * G).sum(), (phi, msg, Y))
    rel = lambda a, b: ((a - b).abs().max() / b.abs().max()).item()
    errs = [rel(D1, D2)] + [rel(a, b) for a, b in zip(g1, g2)]
    assert max(errs) < 1e-4, errs
    print(f"B={B} K={K}: fused == reference (max rel err {max(errs):.1e} over D, dphi, dmsg, dY)")


def check_model():
    from face.model import FACE
    from face.neighbors import padded_graph
    from tools.bench import aspirin_box
    torch.manual_seed(0)
    m = FACE([1, 6, 8], C=32, transport=True, eqmsg=True, factor=True, B=16, K=32).cuda().eval()
    g = padded_graph([aspirin_box(2)], 5.0)
    o1 = m(dict(g, pos=g["pos"].clone()))
    o2 = m.fuse()(dict(g, pos=g["pos"].clone()))
    m.fuse(False)
    dE = (o1["energy"] - o2["energy"]).abs().item(); dF = (o1["forces"] - o2["forces"]).abs().max().item()
    assert dE < 1e-3 and dF < 1e-4, (dE, dF)
    print(f"full model fused vs unfused: |dE| {dE:.1e} eV, max|dF| {dF:.1e} eV/A")


def check_spline():
    from face.model import FACE
    from face.neighbors import padded_graph
    from tools.bench import aspirin_box
    torch.manual_seed(0)
    m = FACE([1, 6, 8], C=32, transport=True, eqmsg=True, factor=True, B=16, K=32).cuda().eval()
    for p in m.parameters():
        p.data += 0.05 * torch.randn_like(p)
    g = padded_graph([aspirin_box(2)], 5.0)
    o1 = m(dict(g, pos=g["pos"].clone()))
    o2 = m.compile_radial()(dict(g, pos=g["pos"].clone()))
    m.compile_radial(on=False)
    dE = (o1["energy"] - o2["energy"]).abs().item() / len(g["Z"])
    dF = ((o1["forces"] - o2["forces"]).norm() / o1["forces"].norm()).item()
    # float32 round-off (different evaluation order); in float64 the interpolation error is 6e-10 at 4096 pts
    assert dE < 1e-5 and dF < 1e-3, (dE, dF)
    print(f"compiled radial vs MLP: |dE| {dE:.1e} eV/atom, rel |dF| {dF:.1e}")


if __name__ == "__main__":
    check_spline()
    check_model()
    check()
    check(B=32, K=32)
    check(B=8, K=16)
