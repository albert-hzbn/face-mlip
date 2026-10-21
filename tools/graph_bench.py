"""CUDA-graph capture of energy+forces for fixed shapes (MD-style inference): replay vs eager."""
import sys, time
import torch
sys.path.insert(0, ".")
from face import safety
from face.model import FACE, unit_parameterise
from face.neighbors import padded_graph
from tools.bench import aspirin_box

safety.install()
for side in (2, 4, 6):
    at = aspirin_box(side)
    m = unit_parameterise(FACE([1, 6, 8], C=96, transport=True, eqmsg=True, factor=True, B=16, K=32).cuda().eval())
    m.fuse().compile_radial()
    g = padded_graph([at], m.rc); g["nstr"] = 1
    static_pos = g["pos"].clone()

    def step():
        gg = dict(g, pos=static_pos.clone().requires_grad_(True))
        return m(gg)["forces"]

    def timeit(f, n=20):
        for _ in range(3): f()
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(n): f()
        torch.cuda.synchronize(); return (time.perf_counter() - t) / n

    t_eager = timeit(step)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): step()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        F_static = step()
    F_ref = step()
    t_graph = timeit(graph.replay)
    err = (F_static - F_ref).abs().max().item()
    print(f"{len(at):5d} atoms: eager {1e6*t_eager/len(at):6.1f} us/atom   CUDA graph {1e6*t_graph/len(at):6.1f} us/atom   "
          f"({t_eager/t_graph:.2f}x)  max|F_graph - F_eager| {err:.1e}", flush=True)
