"""Cost of the physics operators versus system size (flagship, fused kernels + spline radial, periodic aspirin boxes).

usage: python tools/bench_operators.py   -> runs/bench_operators.json
For each size the local model and the model with each operator (latent charges with SPME in coulomb mode,
dispersion with a 10 A pair list, ZBL) and with all three are timed in interleaved rounds (medians); the overhead is
relative to the local model in the same rounds. All variants use the same graph (incl. long-range pair list).
"""
import sys, time, json, statistics
import numpy as np
import torch
sys.path.insert(0, ".")
from face import safety
from face.model import FACE, unit_parameterise
from face.datasets import GraphSet
from tools.bench import aspirin_box

FLAG = dict(C=96, transport=True, eqmsg=True, factor=True, K=32, B=16, rc=4.5)
VARIANTS = {"local": (), "electrostatics": ("coulomb",), "dispersion": ("disp",), "ZBL": ("zbl",),
            "all three": ("coulomb", "disp", "zbl")}


def make(g, terms):
    m = FACE([1, 6, 8], terms=terms, **FLAG).cuda().eval()
    unit_parameterise(m)
    m.fuse(True); m.compile_radial()

    def f():
        gg = dict(g); gg["pos"] = g["pos"].clone()
        return m(gg)["forces"]
    return f


if __name__ == "__main__":
    safety.install()
    torch.set_default_dtype(torch.float32)
    out = {}
    for side in (3, 4, 5, 6):
        at = aspirin_box(side)
        at.info.update(energy=0.0, charge=0.0); at.arrays["forces"] = np.zeros((len(at), 3))
        g = GraphSet([at], FLAG["rc"], rc_long=10.0).batch([0])
        fns = {k: make(g, t) for k, t in VARIANTS.items()}
        times = {k: [] for k in fns}
        for f in fns.values():
            for _ in range(3):
                f()
        for _ in range(5):
            for k, f in fns.items():
                torch.cuda.synchronize(); t = time.perf_counter()
                for _ in range(5):
                    f()
                torch.cuda.synchronize(); times[k].append((time.perf_counter() - t) / 5)
        ref = statistics.median(times["local"])
        row = {k: 100 * (statistics.median(v) / ref - 1) for k, v in times.items() if k != "local"}
        out[len(at)] = row
        print(len(at), {k: f"{v:+.1f}%" for k, v in row.items()}, flush=True)
        del fns; torch.cuda.empty_cache()
    json.dump(out, open("runs/bench_operators.json", "w"), indent=1)
