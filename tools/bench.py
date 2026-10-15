"""Inference throughput: energy + forces, FACE vs MACE, same structures, same GPU.

Systems: periodic boxes of randomly oriented aspirin molecules at ~1.4 g/cm^3.
Reports microseconds per atom per step (neighbour list construction excluded for both).
"""
import sys, time, json, argparse
import numpy as np
import torch
from ase import Atoms
sys.path.insert(0, ".")
from face.data import load_rmd17
from face.model import FACE
from face.neighbors import padded_graph
from scipy.spatial.transform import Rotation
from face import safety


def aspirin_box(nside, seed=0):
    Z, R, *_ = load_rmd17("aspirin")
    rng = np.random.default_rng(seed)
    a = 5.97
    pos, num = [], []
    for ix in range(nside):
        for iy in range(nside):
            for iz in range(nside):
                x = R[rng.integers(len(R))]
                x = (x - x.mean(0)) @ Rotation.random(random_state=rng.integers(1 << 31)).as_matrix().T
                pos.append(x + a * np.array([ix, iy, iz])); num.append(Z)
    return Atoms(numbers=np.concatenate(num), positions=np.concatenate(pos),
                 cell=np.eye(3) * a * nside, pbc=True)


def timeit(fn, n=20, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n


def bench_face(at, dtype, **kw):
    torch.set_default_dtype(dtype)
    fused = kw.pop("fused", False)
    m = FACE([1, 6, 8], **kw).to("cuda").fuse(fused)
    m.eval()
    if kw.get("terms"):
        # physics operators need the cell / total charge / long-range pair list
        from face.datasets import GraphSet
        a = at.copy(); a.info.update(energy=0.0, charge=0.0); a.arrays["forces"] = np.zeros((len(a), 3))
        g = GraphSet([a], m.rc, dtype=dtype, rc_long=10.0 if "disp" in kw["terms"] else None).batch([0])
    else:
        g = padded_graph([at], m.rc, dtype=dtype)

    def f():
        gg = dict(g); gg["pos"] = g["pos"].clone()
        return m(gg, forces=True)["forces"]
    t = timeit(f)
    mem = torch.cuda.max_memory_allocated() / 2**20
    return t, mem


def bench_mace(at, dtype, channels=256, L=2, max_ell=3, correlation=3, layers=2):
    from e3nn import o3
    from mace import modules, data as mdata, tools
    from mace.tools import torch_geometric
    torch.set_default_dtype(dtype)
    hidden = " + ".join(f"{channels}x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(L + 1))
    z_table = tools.AtomicNumberTable([1, 6, 8])
    model = modules.ScaleShiftMACE(
        r_max=5.0, num_bessel=8, num_polynomial_cutoff=5, max_ell=max_ell,
        interaction_cls=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        num_interactions=layers, num_elements=3, hidden_irreps=o3.Irreps(hidden),
        MLP_irreps=o3.Irreps("16x0e"), atomic_energies=np.zeros(3), avg_num_neighbors=20.0,
        atomic_numbers=[1, 6, 8], correlation=correlation, gate=torch.nn.functional.silu,
        atomic_inter_scale=1.0, atomic_inter_shift=0.0, radial_MLP=[64, 64, 64],
    ).to("cuda")
    model.eval()
    cfg = mdata.config_from_atoms(at)
    ad = mdata.AtomicData.from_config(cfg, z_table=z_table, cutoff=5.0)
    batch = next(iter(torch_geometric.dataloader.DataLoader([ad], batch_size=1))).to("cuda")
    d = batch.to_dict()

    def f():
        dd = dict(d); dd["positions"] = d["positions"].clone()
        return model(dd, compute_force=True, training=False)["forces"]
    t = timeit(f)
    mem = torch.cuda.max_memory_allocated() / 2**20
    return t, mem


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sides", default="2,4,6")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--models", default="face,mace_small,mace_paper")
    p.add_argument("--C", type=int, default=64)
    p.add_argument("--v3", type=int, default=1)
    p.add_argument("--factor", type=int, default=1)
    p.add_argument("--B", type=int, default=16)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--kdir", type=int, default=16)
    p.add_argument("--fused", type=int, default=0)
    p.add_argument("--terms", default="")
    a = p.parse_args()
    safety.install()
    dead = set()
    dtype = getattr(torch, a.dtype)
    res = []
    for s in map(int, a.sides.split(",")):
        at = aspirin_box(s)
        n = len(at)
        for name in a.models.split(","):
            if name in dead:
                continue
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            try:
                if name == "face":
                    t, mem = bench_face(at, dtype, C=a.C, transport=bool(a.v3), eqmsg=bool(a.v3),
                                           factor=bool(a.factor), B=a.B, K=a.K, K_dir=a.kdir, fused=bool(a.fused),
                                           terms=tuple(t for t in a.terms.split(",") if t))
                elif name == "mace_small":
                    t, mem = bench_mace(at, dtype, channels=64, L=1, max_ell=3, correlation=3)
                elif name == "mace_paper":   # rMD17 setting of Batatia et al. 2022
                    t, mem = bench_mace(at, dtype, channels=256, L=2, max_ell=3, correlation=3)
                r = {"model": name, "atoms": n, "ms_per_step": 1e3 * t, "us_per_atom": 1e6 * t / n,
                     "peak_MB": mem}
            except (torch.cuda.OutOfMemoryError, RuntimeError) as ex:
                if "out of memory" not in str(ex) and not isinstance(ex, torch.cuda.OutOfMemoryError):
                    raise
                r = {"model": name, "atoms": n, "OOM": True}
                dead.add(name)   # never try a larger system for this model
            print(json.dumps(r), flush=True)
            res.append(r)
    json.dump(res, open(f"runs/bench_{a.dtype}.json", "w"), indent=1)
