import sys, math
import torch
sys.path.insert(0, ".")
from face.model import FACE

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)


def graph(pos, Z):
    n = len(Z)
    full = torch.arange(n)[None].expand(n, n)
    off = ~torch.eye(n, dtype=torch.bool)
    return {"pos": pos.clone(), "Z": Z, "nbr": full[off].view(n, n - 1),
            "shift": torch.zeros(n, n - 1, 3), "mask": torch.ones(n, n - 1, dtype=torch.bool),
            "batch": torch.zeros(n, dtype=torch.long)}


def rand_rot():
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    return q * torch.sign(torch.det(q))


def run(terms=(), transport=False, eqmsg=False, factor=False, saturate=False, standardise=False, calib=False):
    n = 12
    Z = torch.tensor([1, 6, 8, 7] * 3)
    pos = torch.randn(n, 3) * 1.6
    m = FACE([1, 6, 7, 8], C=16, terms=terms, transport=transport, eqmsg=eqmsg, factor=factor)
    m.set_saturation(saturate, 0.5)
    m.set_standardise(standardise)
    m.set_calib(calib)
    for p in m.parameters():
        p.data += 0.1 * torch.randn_like(p)
    o = m(graph(pos, Z))
    E, Fo = o["energy"], o["forces"]
    # rotation (+ reflection) + translation
    for Q in [rand_rot(), -rand_rot()]:
        o2 = m(graph(pos @ Q.T + torch.randn(3), Z))
        assert torch.allclose(o2["energy"], E, atol=1e-9), (o2["energy"], E)
        assert torch.allclose(o2["forces"], Fo @ Q.T, atol=1e-8)
    # permutation
    perm = torch.randperm(n)
    o3 = m(graph(pos[perm], Z[perm]))
    assert torch.allclose(o3["energy"], E, atol=1e-9)
    assert torch.allclose(o3["forces"], Fo[perm], atol=1e-8)
    # finite-difference forces
    h = 1e-5
    for (i, c) in [(0, 0), (3, 2), (7, 1)]:
        pp, pm = pos.clone(), pos.clone()
        pp[i, c] += h; pm[i, c] -= h
        fd = -(m(graph(pp, Z), forces=False)["energy"] - m(graph(pm, Z), forces=False)["energy"]) / (2 * h)
        assert abs(fd.item() - Fo[i, c].item()) < 1e-6 * (1 + abs(fd.item())), (fd.item(), Fo[i, c].item())
    # second derivatives (force training uses create_graph=True) must be finite
    gr = graph(pos, Z); oo = m(gr, create_graph=True)
    (oo["forces"] ** 2).sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None), "non-finite 2nd derivative"
    m.zero_grad()
    # C1 continuity across the cutoff: energy and force just inside / just outside r_c
    p2 = torch.zeros(3, 3); p2[1, 0] = 1.0
    vals = []
    for x in (m.rc - 1e-6, m.rc + 1e-6):
        p2[2] = torch.tensor([x, 0.0, 0.0])
        o = m(graph(p2, torch.tensor([6, 1, 8])))
        vals.append((o["energy"].item(), o["forces"][2, 0].item()))
    dE, dF = abs(vals[0][0] - vals[1][0]), abs(vals[0][1] - vals[1][1])
    assert dE < 2e-6 * (abs(vals[0][1]) + 1) and dF < 1e-5, (dE, dF)   # dE ~ slope * 2 delta
    jumps = dF
    print(f"terms={terms} transport={transport} eqmsg={eqmsg} factor={factor} sat={saturate} std={standardise} calib={calib}: OK  E={E.item():.4f}  max force step across rc={jumps:.2e}")


if __name__ == "__main__":
    run()
    run(("zbl", "coulomb"))
    run(transport=True)
    run(transport=True, eqmsg=True)
    run(("zbl", "qeq"), transport=True, eqmsg=True)
    run(("zbl", "coulomb"), transport=True, eqmsg=True, factor=True)
    run(("zbl", "coulomb"), transport=True, eqmsg=True, factor=True, saturate=True)
    run(("zbl", "qeq"), transport=True, eqmsg=True, factor=True, standardise=True)
    run(("zbl", "coulomb"), transport=True, eqmsg=True, factor=True, calib=True)
