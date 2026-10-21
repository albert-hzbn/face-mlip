"""Load-robust training-step cost: FACE vs MACE, interleaved rounds, medians.
One step = forward + forces (create_graph) + E/F loss + backward (second derivatives) + optimiser step,
batch of 5 aspirin structures (the training setting used throughout)."""
import sys, time, statistics
import numpy as np
import torch
sys.path.insert(0, ".")
from face import safety
from face.data import load_rmd17, MoleculeSet
from face.model import FACE, unit_parameterise


def face_step(**kw):
    Z, R, E, Fo, idx = load_rmd17("aspirin")
    ds = MoleculeSet(Z, R, E - E.mean(), Fo, "cuda", torch.float32)
    m = unit_parameterise(FACE(sorted(set(Z.tolist())), **kw).cuda())
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, amsgrad=True)
    tr = idx["train"]; rng = np.random.default_rng(0)

    def step():
        g = ds.batch(rng.choice(tr, 5))
        o = m(g, create_graph=True)
        loss = (((o["energy"] - g["E"]) / 21) ** 2).mean() + 1000 * ((o["forces"] - g["F"]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return step, sum(p.numel() for p in m.parameters())


def mace_step(channels):
    from e3nn import o3
    from mace import modules, data as mdata, tools
    from mace.tools import torch_geometric
    from ase import Atoms
    Z, R, E, Fo, idx = load_rmd17("aspirin")
    hidden = f"{channels}x0e + {channels}x1o"
    model = modules.ScaleShiftMACE(
        r_max=5.0, num_bessel=8, num_polynomial_cutoff=5, max_ell=3,
        interaction_cls=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        num_interactions=2, num_elements=3, hidden_irreps=o3.Irreps(hidden), MLP_irreps=o3.Irreps("16x0e"),
        atomic_energies=np.zeros(3), avg_num_neighbors=14.4, atomic_numbers=[1, 6, 8], correlation=3,
        gate=torch.nn.functional.silu, atomic_inter_scale=1.0, atomic_inter_shift=0.0,
        radial_MLP=[64, 64, 64]).cuda()
    zt = tools.AtomicNumberTable([1, 6, 8])
    tr = idx["train"]; rng = np.random.default_rng(0)
    data = [mdata.AtomicData.from_config(mdata.config_from_atoms(Atoms(numbers=Z, positions=R[i])), z_table=zt,
                                         cutoff=5.0) for i in tr[:200]]
    Ft = torch.tensor(Fo[tr[:200]], dtype=torch.float32, device="cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, amsgrad=True)

    def step():
        sel = rng.choice(200, 5)
        b = next(iter(torch_geometric.dataloader.DataLoader([data[k] for k in sel], batch_size=5))).to("cuda").to_dict()
        o = model(b, training=True, compute_force=True)
        loss = ((o["forces"] - Ft[sel].reshape(-1, 3)) ** 2).mean() * 1000 + (o["energy"] ** 2).mean() * 0
        opt.zero_grad(); loss.backward(); opt.step()
    return step, sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    safety.install()
    torch.set_default_dtype(torch.float32)
    cfgs = {"MACE-small": lambda: mace_step(64), "MACE-medium": lambda: mace_step(128),
            "FACE flagship": lambda: face_step(C=96, transport=True, eqmsg=True, factor=True, B=16, K=32, rc=4.5),
            "FACE-CP C96": lambda: face_step(C=96, transport=True, eqmsg=True, factor=True, B=16, cp=True),
            "FACE-CP C160": lambda: face_step(C=160, transport=True, eqmsg=True, factor=True, B=16, cp=True)}
    steps, params, times = {}, {}, {}
    for k, make in cfgs.items():
        steps[k], params[k] = make(); times[k] = []
        for _ in range(5):
            steps[k]()
    for r in range(7):
        for k, f in steps.items():
            torch.cuda.synchronize(); t = time.perf_counter()
            for _ in range(10):
                f()
            torch.cuda.synchronize(); times[k].append((time.perf_counter() - t) / 10)
    ref = statistics.median(times["MACE-small"])
    for k in steps:
        med = statistics.median(times[k])
        print(f"{k:16s} params {params[k]:>9,d}   train step {1e3*med:6.1f} ms   relative to MACE-small {med/ref:4.2f}x", flush=True)
