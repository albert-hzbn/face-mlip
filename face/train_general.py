"""Train FACE on a generic structure set (variable composition / charge / PBC)."""
import argparse, json, time, copy, os
import numpy as np
import torch

from . import safety
from .datasets import read_runner, GraphSet, fit_E0
from .model import FACE, unit_parameterise


def evaluate(model, ds, idx, bs=32):
    model.eval()
    dE, dF, dq = [], [], []
    for k in range(0, len(idx), bs):
        g = ds.batch(idx[k:k + bs])
        o = model(g)
        dE.append(((o["energy"] - g["E"]) / g["natoms"]).detach().double().cpu().numpy())
        dF.append((o["forces"] - g["F"]).detach().double().cpu().numpy().ravel())
    dE, dF = np.concatenate(dE), np.concatenate(dF)
    return {"E_MAE_meV_atom": 1e3 * np.abs(dE).mean(), "E_RMSE_meV_atom": 1e3 * np.sqrt((dE ** 2).mean()),
            "F_MAE_meV_A": 1e3 * np.abs(dF).mean(), "F_RMSE_meV_A": 1e3 * np.sqrt((dF ** 2).mean())}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--n_train", type=int, default=900)
    p.add_argument("--n_val", type=int, default=100)
    p.add_argument("--n_test", type=int, default=1000)
    p.add_argument("--C", type=int, default=64)
    p.add_argument("--rc", type=float, default=5.0)
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--wF", type=float, default=100.0)
    p.add_argument("--terms", default="")
    p.add_argument("--transport", type=int, default=1)
    p.add_argument("--saturate", type=int, default=0)
    p.add_argument("--standardise", type=int, default=0)
    p.add_argument("--momnorm", type=int, default=0)
    p.add_argument("--calib", type=int, default=0)
    p.add_argument("--calib_ln", type=int, default=0)
    p.add_argument("--eq_linear", type=int, default=0)
    p.add_argument("--max_body", type=int, default=5)
    p.add_argument("--unit", type=int, default=1)
    p.add_argument("--cp", type=int, default=0)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--elem_proj", type=int, default=0)
    p.add_argument("--poly", type=int, default=0)
    p.add_argument("--t3", type=int, default=0)
    p.add_argument("--eqmsg", type=int, default=1)
    p.add_argument("--factor", type=int, default=1)
    p.add_argument("--B", type=int, default=16)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="gen")
    a = p.parse_args(argv)
    safety.install()
    torch.manual_seed(a.seed)
    torch.set_default_dtype(torch.float32)
    atoms = read_runner(a.data)
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(atoms))
    tr, va = perm[:a.n_train], perm[a.n_train:a.n_train + a.n_val]
    te = perm[a.n_train + a.n_val:a.n_train + a.n_val + a.n_test]
    E0 = fit_E0([atoms[i] for i in tr])
    ds = GraphSet(atoms, a.rc, E0=E0)
    elements = sorted(E0)
    terms = [t for t in a.terms.split(",") if t]
    model = FACE(elements, C=a.C, rc=a.rc, n_passes=a.passes, terms=terms,
                   transport=bool(a.transport), eqmsg=bool(a.eqmsg),
                   factor=bool(a.factor), B=a.B, K=a.K, cp=bool(a.cp)).cuda()
    model.set_saturation(bool(a.saturate))
    model.set_standardise(bool(a.standardise))
    model.set_momnorm(bool(a.momnorm))
    model.set_calib(bool(a.calib), keep_ln=bool(a.calib_ln))
    model.set_eq_linear(bool(a.eq_linear))
    model.set_max_body(a.max_body)
    model.set_elem_proj(bool(a.elem_proj))
    model.set_poly_readout(bool(a.poly))
    model.set_transport3(bool(a.t3))
    if a.unit:
        unit_parameterise(model)
    with torch.no_grad():
        Fs = np.concatenate([atoms[i].arrays["forces"].ravel() for i in tr])
        model.scale.fill_(float(Fs.std()))
        g = ds.batch(tr[:64])
        model.avg_nb.fill_(float(g["mask"].sum(1).float().mean()))
    print(f"{len(atoms)} structures, elements {elements}, terms {terms}, params "
          f"{sum(p.numel() for p in model.parameters())}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, amsgrad=True, weight_decay=a.wd)
    spe = max(1, len(tr) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * spe, pct_start=0.05,
                                                div_factor=10, final_div_factor=100)
    ema = copy.deepcopy(model)
    swa_start = int(0.75 * a.epochs)
    best, t0, log = (1e18, None), time.time(), []
    rec = {"bad": 0, "lr_scale": 1.0, "restores": 0}

    def on_bad_step():
        """Recovery: after 20 consecutive non-finite steps restore weights from the EMA model,
        reset the optimiser state and halve the learning-rate multiplier (training never freezes)."""
        rec["bad"] += 1
        if rec["bad"] >= 20:
            with torch.no_grad():
                for pm, pe in zip(model.parameters(), ema.parameters()):
                    pm.copy_(pe)
            opt.state.clear()
            rec["lr_scale"] *= 0.5
            rec["restores"] += 1
            rec["bad"] = 0
            print(f"[recovery] restored EMA weights, lr multiplier -> {rec['lr_scale']}", flush=True)

    for ep in range(a.epochs):
        model.train()
        wE, wF = (1.0, a.wF) if ep < swa_start else (100.0, a.wF / 10)
        order = rng.permutation(tr)
        for k in range(spe):
            g = ds.batch(order[k * a.bs:(k + 1) * a.bs])
            o = model(g, create_graph=True)
            lE = (((o["energy"] - g["E"]) / g["natoms"]) ** 2).mean()
            lF = ((o["forces"] - g["F"]) ** 2).mean()
            loss = wE * lE + wF * lF
            if not torch.isfinite(loss):          # guard: skip non-finite steps instead of poisoning weights
                skipped = globals().get("_skipped", 0) + 1; globals()["_skipped"] = skipped
                print(f"non-finite loss, step skipped (total {skipped})", flush=True)
                on_bad_step()
                continue
            opt.zero_grad(set_to_none=True); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            if not torch.isfinite(gn):            # finite loss but non-finite gradient: skip, keep weights clean
                print(f"non-finite gradient (norm {gn.item()}, loss {loss.item():.3e}), step skipped", flush=True)
                opt.zero_grad(set_to_none=True)
                on_bad_step()
                continue
            rec["bad"] = 0
            opt.step(); sched.step()
            if rec["lr_scale"] != 1.0:
                for pg in opt.param_groups:
                    pg["lr"] *= rec["lr_scale"]
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(0.99).add_(pm.detach(), alpha=0.01)
                for be, bm in zip(ema.buffers(), model.buffers()):   # calibration scales etc.
                    be.copy_(bm)
        if ep % 10 == 0 or ep == a.epochs - 1:
            v = evaluate(ema, ds, va)
            v.update(epoch=ep, time=time.time() - t0); log.append(v)
            print(json.dumps({k: round(x, 3) if isinstance(x, float) else x for k, x in v.items()}), flush=True)
            score = v["E_RMSE_meV_atom"] + v["F_RMSE_meV_A"] / 10
            if ep >= swa_start and score < best[0]:
                best = (score, copy.deepcopy(ema.state_dict()))
    if best[1] is not None:
        ema.load_state_dict(best[1])
    res = evaluate(ema, ds, te)
    print("TEST", json.dumps({k: round(x, 3) for k, x in res.items()}), flush=True)
    os.makedirs("runs", exist_ok=True)
    name = os.path.basename(os.path.dirname(a.data))
    json.dump({"args": vars(a), "test": res, "log": log}, open(f"runs/{a.tag}_{name}.json", "w"), indent=1)
    torch.save({"state": ema.state_dict(), "args": vars(a), "E0": E0}, f"runs/{a.tag}_{name}.pt")


if __name__ == "__main__":
    main()
