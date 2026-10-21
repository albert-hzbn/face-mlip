import argparse, json, time, copy, os
import numpy as np
import torch

from .data import load_rmd17, MoleculeSet
from .model import FACE, unit_parameterise
from . import safety


def evaluate(model, ds, idx, bs=50):
    model.eval()
    eE, eF, nF = 0.0, 0.0, 0
    for k in range(0, len(idx), bs):
        g = ds.batch(idx[k:k + bs])
        out = model(g, forces=True)
        eE += (out["energy"] - g["E"]).abs().sum().item()
        eF += (out["forces"] - g["F"]).abs().sum().item()
        nF += g["F"].numel()
    return 1000 * eE / len(idx), 1000 * eF / nF


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mol", default="aspirin")
    p.add_argument("--C", type=int, default=64)
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--rc", type=float, default=5.0)
    p.add_argument("--nbasis", type=int, default=8)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--bs", type=int, default=5)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--wF", type=float, default=1000.0)
    p.add_argument("--wE", type=float, default=1.0)
    p.add_argument("--swa_frac", type=float, default=0.25)
    p.add_argument("--wE2", type=float, default=1000.0)   # loss weights in the final (energy) phase
    p.add_argument("--wF2", type=float, default=100.0)
    p.add_argument("--ema", type=float, default=0.99)
    p.add_argument("--terms", default="")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="run")
    p.add_argument("--norm", default="layer")
    p.add_argument("--transport", type=int, default=0)
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
    p.add_argument("--aug", default="")             # teacher-labelled npz (ensemble distillation), train only
    p.add_argument("--dft_rep", type=int, default=4)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--init", default="")             # start from a checkpoint (e.g. fine-tune a distilled model)
    p.add_argument("--eqmsg", type=int, default=0)
    p.add_argument("--factor", type=int, default=0)
    p.add_argument("--B", type=int, default=16)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--kdir", type=int, default=16)
    p.add_argument("--radial_hidden", type=int, default=64)
    p.add_argument("--max_test", type=int, default=1000)
    p.add_argument("--n_train", type=int, default=0, help="use only the first n of the shuffled training structures (learning curves)")
    a = p.parse_args(argv)
    safety.install()
    if os.environ.get("FACE_ANOMALY"):
        torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)          # batch order is part of the run: make it reproducible
    dev = "cuda"
    dtype = getattr(torch, a.dtype)
    torch.set_default_dtype(dtype)

    Z, R, E, Fo, idx = load_rmd17(a.mol)
    if a.n_train:
        idx["train"] = idx["train"][:a.n_train]       # nested subsets: same validation and test sets
    # composition-constant reference removed in float64 before casting
    Eref = E[idx["train"]].mean()
    train_idx = np.asarray(idx["train"])
    if a.aug:
        # ensemble distillation: append teacher-labelled snapshots (never touches val/test indices)
        parts = [np.load(f) for f in a.aug.split(",")]          # several teacher batches, comma-separated
        assert all((p_["Z"] == Z).all() for p_ in parts)
        aug = {k: np.concatenate([p_[k] for p_ in parts]) for k in ("R", "E", "F")}
        n0 = len(E)
        R = np.concatenate([R, aug["R"]]); E = np.concatenate([E, aug["E"]]); Fo = np.concatenate([Fo, aug["F"]])
        train_idx = np.concatenate([np.tile(train_idx, a.dft_rep), n0 + np.arange(len(aug["E"]))])
        print(f"distillation: {len(aug['E'])} teacher-labelled + {len(idx['train'])}x{a.dft_rep} DFT configurations", flush=True)
    ds = MoleculeSet(Z, R, E - Eref, Fo, dev, dtype)
    elements = sorted(set(Z.tolist()))
    terms = [t for t in a.terms.split(",") if t]
    model = FACE(elements, C=a.C, n_basis=a.nbasis, rc=a.rc, n_passes=a.passes, terms=terms,
                   radial_hidden=a.radial_hidden, K_dir=a.kdir, norm=a.norm,
                   transport=bool(a.transport), eqmsg=bool(a.eqmsg),
                   factor=bool(a.factor), B=a.B, K=a.K, cp=bool(a.cp)).to(dev)
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
        model.scale.fill_(float(ds.F[idx["train"]].std()))
        pos = ds.R[idx["train"][:100]]
        d = (pos[:, :, None] - pos[:, None]).norm(dim=-1)
        model.avg_nb.fill_(float(((d < a.rc).sum(-1) - 1).double().mean()))
    if a.init:
        ck = torch.load(a.init, weights_only=False, map_location=dev)
        missing, unexpected = model.load_state_dict(ck["state"], strict=False)
        # buffers and parameters added after older checkpoints keep their neutral defaults (scales 1, product gates 1,
        # rank-3 transport map unused unless --t3), so the loaded model computes the same function
        allowed = ("inv_scale", "calibrated", "mom_scale", "eq_scale", "pqs_gate.weight", "tmix3.weight")
        assert not unexpected and all(k.endswith(allowed) for k in missing), (missing, unexpected)
        print(f"initialised from {a.init}", flush=True)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"params {nparam}  avg_nb {model.avg_nb.item():.2f}  scale {model.scale.item():.3f}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, amsgrad=True, weight_decay=a.wd)
    steps_per_epoch = len(train_idx) // a.bs
    total = a.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.05,
                                                anneal_strategy="cos", div_factor=10, final_div_factor=100)
    ema = copy.deepcopy(model)
    swa_start = int((1 - a.swa_frac) * a.epochs)
    log, best, t0 = [], (1e9, None), time.time()
    tr = train_idx
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
        wE, wF = (a.wE, a.wF) if ep < swa_start else (a.wE2, a.wF2)
        perm = np.random.permutation(tr)
        tl = 0.0
        for k in range(steps_per_epoch):
            if os.path.exists(f"runs/STOP_{a.tag}"):
                # graceful stop: hard-killing a busy CUDA process triggers WDDM GPU resets on this machine
                torch.save({"state": ema.state_dict(), "args": vars(a), "elements": elements, "Eref": Eref,
                            "epoch": ep - 1}, f"runs/{a.tag}_{a.mol}_last.pt")
                print(f"stop file runs/STOP_{a.tag} found: saved runs/{a.tag}_{a.mol}_last.pt, exiting cleanly", flush=True)
                return
            g = ds.batch(perm[k * a.bs:(k + 1) * a.bs])
            out = model(g, forces=True, create_graph=True)
            n = ds.N
            lE = (((out["energy"] - g["E"]) / n) ** 2).mean()
            lF = ((out["forces"] - g["F"]) ** 2).mean()
            loss = wE * lE + wF * lF
            if not torch.isfinite(loss):          # guard: skip non-finite steps instead of poisoning weights
                skipped = globals().get("_skipped", 0) + 1; globals()["_skipped"] = skipped
                print(f"non-finite loss, step skipped (total {skipped})", flush=True)
                on_bad_step()
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            if not torch.isfinite(gn):            # finite loss but non-finite gradient: skip, keep weights clean
                print(f"non-finite gradient (norm {gn.item()}, loss {loss.item():.3e}) at epoch {ep} step {k}, step skipped", flush=True)
                if os.environ.get("FACE_DUMP_NAN"):
                    torch.save({"state": model.state_dict(), "batch": perm[k * a.bs:(k + 1) * a.bs]}, "runs/nan_state.pt")
                    raise SystemExit("dumped runs/nan_state.pt")
                opt.zero_grad(set_to_none=True)
                on_bad_step()
                continue
            rec["bad"] = 0
            opt.step()
            sched.step()
            if rec["lr_scale"] != 1.0:
                for pg in opt.param_groups:
                    pg["lr"] *= rec["lr_scale"]
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(a.ema).add_(pm.detach(), alpha=1 - a.ema)
                for be, bm in zip(ema.buffers(), model.buffers()):   # calibration scales etc.
                    be.copy_(bm)
            tl += loss.item()
        if ep % a.log_every == 0 or ep == a.epochs - 1:
            vE, vF = evaluate(ema, ds, idx["val"])
            log.append({"epoch": ep, "loss": tl / steps_per_epoch, "val_E": vE, "val_F": vF,
                        "time": time.time() - t0})
            print(json.dumps(log[-1]), flush=True)
            # periodic EMA checkpoint so a crash on a shared GPU costs at most one logging interval (resume via --init)
            os.makedirs("runs", exist_ok=True)
            torch.save({"state": ema.state_dict(), "args": vars(a), "elements": elements, "Eref": Eref, "epoch": ep},
                       f"runs/{a.tag}_{a.mol}_last.pt")
            score = vE + vF if ep >= swa_start else vF
            if ep >= swa_start and score < best[0]:
                best = (score, copy.deepcopy(ema.state_dict()))
    if best[1] is not None:
        ema.load_state_dict(best[1])
    tE, tF = evaluate(ema, ds, idx["test"][:a.max_test])
    res = {"args": vars(a), "params": nparam, "test_E_meV": tE, "test_F_meV_A": tF,
           "train_time_s": time.time() - t0, "log": log}
    print(f"TEST {a.mol}: E {tE:.3f} meV  F {tF:.3f} meV/A", flush=True)
    os.makedirs("runs", exist_ok=True)
    with open(f"runs/{a.tag}_{a.mol}.json", "w") as f:
        json.dump(res, f, indent=1)
    torch.save({"state": ema.state_dict(), "args": vars(a), "elements": elements, "Eref": Eref},
               f"runs/{a.tag}_{a.mol}.pt")


if __name__ == "__main__":
    main()
