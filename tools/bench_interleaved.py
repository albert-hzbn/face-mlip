"""Load-robust speed ratio: models are timed in interleaved rounds (A,B,C,A,B,C,...) and medians reported,
so background GPU load affects all models equally. Energy + forces, aspirin box (~1.4 g/cm^3)."""
import sys, time, json, statistics, argparse
import numpy as np
import torch
sys.path.insert(0, ".")
from face import safety
from face.model import FACE, unit_parameterise
from face.neighbors import padded_graph
from tools.bench import aspirin_box


def face_fn(at, fused, compiled, t3=False, **kw):
    m = FACE([1, 6, 8], **kw).cuda().eval()
    m.set_transport3(t3)
    unit_parameterise(m)
    m.fuse(fused)
    if compiled:
        m.compile_radial()
    g = padded_graph([at], m.rc)
    return lambda: m(dict(g, pos=g["pos"].clone()))["forces"]


def mace_fn(at, channels=64, L=1):
    from e3nn import o3
    from mace import modules, data as mdata, tools
    from mace.tools import torch_geometric
    hidden = " + ".join(f"{channels}x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(L + 1))
    model = modules.ScaleShiftMACE(
        r_max=5.0, num_bessel=8, num_polynomial_cutoff=5, max_ell=3,
        interaction_cls=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        num_interactions=2, num_elements=3, hidden_irreps=o3.Irreps(hidden), MLP_irreps=o3.Irreps("16x0e"),
        atomic_energies=np.zeros(3), avg_num_neighbors=20.0, atomic_numbers=[1, 6, 8], correlation=3,
        gate=torch.nn.functional.silu, atomic_inter_scale=1.0, atomic_inter_shift=0.0,
        radial_MLP=[64, 64, 64]).cuda().eval()
    ad = mdata.AtomicData.from_config(mdata.config_from_atoms(at), z_table=tools.AtomicNumberTable([1, 6, 8]), cutoff=5.0)
    d = next(iter(torch_geometric.dataloader.DataLoader([ad], batch_size=1))).to("cuda").to_dict()
    return lambda: model(dict(d, positions=d["positions"].clone()), compute_force=True)["forces"]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--side", type=int, default=6)
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--only", default="")
    a = p.parse_args()
    safety.install()
    at = aspirin_box(a.side)
    base = dict(C=96, transport=True, eqmsg=True, factor=True, K=32)
    models = {
        "MACE-small (64x0e+64x1o)": lambda: mace_fn(at),
        "MACE-medium (128x0e+128x1o)": lambda: mace_fn(at, channels=128),
        "FACE B16 (PyTorch)": lambda: face_fn(at, False, False, B=16, **base),
        "FACE B16 fused+compiled": lambda: face_fn(at, True, True, B=16, **base),
        "FACE B32 (PyTorch)": lambda: face_fn(at, False, False, B=32, **base),
        "FACE B32 fused+compiled": lambda: face_fn(at, True, True, B=32, **base),
        "FACE B32 kdir32 fused+compiled": lambda: face_fn(at, True, True, B=32, K_dir=32, **base),
        "v8 CP C96 B16 (PyTorch)": lambda: face_fn(at, False, False, B=16, cp=True, **base),
        "v8 CP C96 B16 fused-node": lambda: face_fn(at, True, False, B=16, cp=True, **base),
        "FACE B16 C64 fused+compiled": lambda: face_fn(at, True, True, B=16, **dict(base, C=64)),
        "FACE B16 C96 rc4.5 fused+compiled": lambda: face_fn(at, True, True, B=16, rc=4.5, **base),
        "FACE B16 C96 rc4.5 t3 fused+compiled": lambda: face_fn(at, True, True, t3=True, B=16, rc=4.5, **base),
        "FACE-L C192 B32 kdir32 fused+compiled": lambda: face_fn(at, True, True, B=32, K_dir=32, **dict(base, C=192)),
        "v8 CP C160 B16 fused-node": lambda: face_fn(at, True, False, B=16, cp=True, **dict(base, C=160)),
        "v8 CP C96 B16 fused-node (accuracy)": lambda: face_fn(at, True, False, B=16, cp=True, **base),
        "FACE B16 C64 rc4.5 fused+compiled": lambda: face_fn(at, True, True, B=16, rc=4.5, **dict(base, C=64)),
        "v8 CP C96 B16 compiled": lambda: face_fn(at, False, True, B=16, cp=True, **base),
        "v8 CP C96 B16 kdir32 compiled": lambda: face_fn(at, False, True, B=16, K_dir=32, cp=True, **base),
    }
    if a.only:
        keep = a.only.split(";")
        models = {k: v for k, v in models.items() if k in keep or k.startswith("MACE-small")}
    models = {k: make() for k, make in models.items()}          # build only the selected models
    times = {k: [] for k in models}
    for f in models.values():
        for _ in range(3):
            f()
    for r in range(a.rounds):
        for k, f in models.items():
            torch.cuda.synchronize(); t = time.perf_counter()
            for _ in range(5):
                f()
            torch.cuda.synchronize(); times[k].append((time.perf_counter() - t) / 5)
    ref = statistics.median(times["MACE-small (64x0e+64x1o)"])
    out = {}
    for k, v in times.items():
        med = statistics.median(v)
        out[k] = {"us_per_atom": 1e6 * med / len(at), "speedup_vs_MACE_small": ref / med}
        print(f"{k:36s} {1e6 * med / len(at):7.1f} us/atom   speed-up vs MACE-small {ref / med:5.2f}x", flush=True)
    json.dump({"atoms": len(at), **out}, open("runs/bench_interleaved.json", "w"), indent=1)
