"""Timing checks requested in review (energy + forces, float32, periodic aspirin boxes, interleaved rounds).

usage: python tools/bench_review.py cutoff     -> runs/bench_cutoff.json    (FACE and MACE at r_c = 4.5 and 5.0 A, 1,344 atoms)
       python tools/bench_review.py small      -> runs/bench_168.json       (all Figure-5b models at 168 atoms)
       python tools/bench_review.py profile    -> runs/bench_profile.json   (CUDA-time breakdown by kernel class)
       python tools/bench_review.py qeq        -> runs/bench_qeq.json       (cost of the qeq mode vs system size)
"""
import sys, time, json, statistics, gc
import numpy as np
import torch
sys.path.insert(0, ".")
from face import safety
from face.model import FACE, unit_parameterise
from face.neighbors import padded_graph
from face.datasets import GraphSet
from tools.bench import aspirin_box

FLAG = dict(C=96, transport=True, eqmsg=True, factor=True, K=32, B=16)


def face_fn(at, rc=4.5, fused=True, compiled=True, **kw):
    cfg = dict(FLAG, **kw)
    m = FACE([1, 6, 8], rc=rc, **cfg).cuda().eval()
    unit_parameterise(m)
    m.fuse(fused)
    if compiled:
        m.compile_radial()
    g = padded_graph([at], m.rc)
    return lambda: m(dict(g, pos=g["pos"].clone()))["forces"]


def mace_fn(at, channels=64, L=1, rc=5.0):
    from e3nn import o3
    from mace import modules, data as mdata, tools
    from mace.tools import torch_geometric
    hidden = " + ".join(f"{channels}x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(L + 1))
    model = modules.ScaleShiftMACE(
        r_max=rc, num_bessel=8, num_polynomial_cutoff=5, max_ell=3,
        interaction_cls=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        num_interactions=2, num_elements=3, hidden_irreps=o3.Irreps(hidden), MLP_irreps=o3.Irreps("16x0e"),
        atomic_energies=np.zeros(3), avg_num_neighbors=20.0, atomic_numbers=[1, 6, 8], correlation=3,
        gate=torch.nn.functional.silu, atomic_inter_scale=1.0, atomic_inter_shift=0.0,
        radial_MLP=[64, 64, 64]).cuda().eval()
    ad = mdata.AtomicData.from_config(mdata.config_from_atoms(at), z_table=tools.AtomicNumberTable([1, 6, 8]), cutoff=rc)
    d = next(iter(torch_geometric.dataloader.DataLoader([ad], batch_size=1))).to("cuda").to_dict()
    return lambda: model(dict(d, positions=d["positions"].clone()), compute_force=True)["forces"]


ROUNDS = int(__import__("os").environ.get("BENCH_ROUNDS", 7))


def interleave(fns, n_atoms, rounds=None, calls=5):
    rounds = rounds or ROUNDS
    times = {k: [] for k in fns}
    for f in fns.values():
        for _ in range(3):
            f()
    for _ in range(rounds):
        for k, f in fns.items():
            torch.cuda.synchronize(); t = time.perf_counter()
            for _ in range(calls):
                f()
            torch.cuda.synchronize(); times[k].append((time.perf_counter() - t) / calls)
    out = {}
    for k, v in times.items():
        us = [1e6 * x / n_atoms for x in v]
        out[k] = {"us_per_atom_median": statistics.median(us), "us_per_atom_min": min(us), "us_per_atom_max": max(us)}
    return out


def report(out, ref, path, extra=None):
    r = out[ref]["us_per_atom_median"]
    for k, v in out.items():
        v["speedup_vs_" + ref.replace(" ", "_")] = r / v["us_per_atom_median"]
        print(f"{k:34s} {v['us_per_atom_median']:8.1f} us/atom  (range {v['us_per_atom_min']:.1f}-{v['us_per_atom_max']:.1f})"
              f"  speed-up vs {ref}: {r / v['us_per_atom_median']:5.2f}", flush=True)
    json.dump(dict(out, **(extra or {})), open(path, "w"), indent=1)


def cutoff():
    at = aspirin_box(4)
    fns = {"MACE-small rc5.0": mace_fn(at, 64, rc=5.0), "MACE-small rc4.5": mace_fn(at, 64, rc=4.5),
           "FACE flagship rc4.5": face_fn(at, rc=4.5), "FACE flagship rc5.0": face_fn(at, rc=5.0)}
    report(interleave(fns, len(at)), "MACE-small rc5.0", "runs/bench_cutoff.json", {"atoms": len(at)})


def small():
    at = aspirin_box(2)
    fns = {"MACE-small": mace_fn(at, 64), "MACE 256 ch L=2": mace_fn(at, 256, L=2),
           "FACE flagship": face_fn(at, rc=4.5), "FACE-CP C96": face_fn(at, rc=5.0, compiled=False, cp=True),
           "FACE-CP C160": face_fn(at, rc=5.0, compiled=False, cp=True, C=160)}
    report(interleave(fns, len(at)), "MACE-small", "runs/bench_168.json", {"atoms": len(at)})


def classify(name):
    n = name.lower()
    if "dens_" in n or "factordensity" in n:
        return "fused density kernel"
    if "node_" in n or "nodealgebra" in n:
        return "fused node kernel"
    if "gemm" in n or "cutlass" in n or "sm80_xmma" in n or "ampere" in n or "matmul" in n or "bmm" in n or n in ("aten::mm", "aten::addmm", "aten::linear"):
        return "dense matrix products (GEMM)"
    if "index" in n or "gather" in n or "scatter" in n:
        return "gather / scatter (neighbour indexing)"
    if "reduce" in n or "sum" in n:
        return "reductions"
    return "elementwise and other"


def profile():
    from torch.profiler import profile as prof, ProfilerActivity
    at = aspirin_box(4)
    res = {}
    for name, make in [("FACE flagship", lambda: face_fn(at, rc=4.5)),
                       ("FACE-CP C96", lambda: face_fn(at, rc=5.0, compiled=False, cp=True)),
                       ("MACE-small", lambda: mace_fn(at, 64))]:
        f = make()
        for _ in range(5):
            f()
        torch.cuda.synchronize()
        with prof(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
            for _ in range(10):
                f()
            torch.cuda.synchronize()
        cats = {}
        # On this Windows/WDDM system CUPTI records no separate kernel events; each operator event carries the device
        # time of the kernels it launched itself (self_device_time_total), which is summed by operator class.
        for e in p.key_averages():
            t = e.self_device_time_total
            if t > 0:
                cats[classify(e.key)] = cats.get(classify(e.key), 0.0) + t
        tot = sum(cats.values())
        res[name] = {k: 100 * v / tot for k, v in sorted(cats.items(), key=lambda kv: -kv[1])}
        print(name, {k: f"{v:.0f}%" for k, v in res[name].items()}, flush=True)
        del f; gc.collect(); torch.cuda.empty_cache()
    json.dump(res, open("runs/bench_profile.json", "w"), indent=1)


def qeq():
    out = {}
    for side in (2, 3, 4, 5):
        at = aspirin_box(side)
        at.info.update(energy=0.0, charge=0.0); at.arrays["forces"] = np.zeros((len(at), 3))
        g = GraphSet([at], 4.5, rc_long=None).batch([0])
        fns = {}
        for name, terms in [("local", ()), ("coulomb", ("coulomb",)), ("qeq", ("qeq",))]:
            m = FACE([1, 6, 8], rc=4.5, terms=terms, **FLAG).cuda().eval(); unit_parameterise(m); m.fuse(True); m.compile_radial()
            fns[name] = (lambda m=m: m(dict(g, pos=g["pos"].clone()))["forces"])
        try:
            r = interleave(fns, len(at), rounds=5)
            ref = r["local"]["us_per_atom_median"]
            out[len(at)] = {k: 100 * (v["us_per_atom_median"] / ref - 1) for k, v in r.items() if k != "local"}
            print(len(at), {k: f"{v:+.0f}%" for k, v in out[len(at)].items()}, flush=True)
        except RuntimeError as e:
            if "out of memory" not in str(e):
                raise
            out[len(at)] = "OOM"; print(len(at), "OOM", flush=True)
        del fns; gc.collect(); torch.cuda.empty_cache()
    json.dump(out, open("runs/bench_qeq.json", "w"), indent=1)


if __name__ == "__main__":
    safety.install()
    torch.set_default_dtype(torch.float32)
    {"cutoff": cutoff, "small": small, "profile": profile, "qeq": qeq}[sys.argv[1]]()
