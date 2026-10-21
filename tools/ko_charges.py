"""Latent charges of the long-range FACE models vs the reference Hirshfeld charges of Ko et al. (never trained on).

usage: python tools/ko_charges.py      -> runs/paper_ko_charges.npz + printed statistics
The checkpoints predate the 'unit' flag in the saved arguments; the model is rebuilt with and without unit
parameterisation and the variant that reproduces the stored test error is used.
"""
import sys, json
import numpy as np
import torch
sys.path.insert(0, ".")
from face import safety
from face.datasets import read_runner, GraphSet
from face.model import FACE, unit_parameterise

RUNS = {"Ag3": "ko2_coulomb_Ag_cluster", "C10H2_rc3": "ko_rc3_qeq_Carbon_chain", "C10H2_rc5": "ko2_qeq_Carbon_chain"}


def build(ck, unit):
    a = ck["args"]
    terms = [t for t in a["terms"].split(",") if t]
    m = FACE(sorted(ck["E0"]), C=a["C"], rc=a["rc"], n_passes=a["passes"], terms=terms, transport=bool(a["transport"]),
             eqmsg=bool(a["eqmsg"]), factor=bool(a["factor"]), B=a["B"], K=a["K"]).cuda()
    if unit:
        unit_parameterise(m)
    # backward compatibility: these checkpoints predate the 4 optional moment-norm readout inputs (always zero when
    # moment normalisation is off -> zero-padded columns give the identical function) and later buffers/options
    # (scales = 1, product gates = 1, rank-3 transport unused), which keep their defaults
    st = dict(ck["state"])
    cur = m.state_dict()
    for k, v in st.items():
        if k in cur and v.shape != cur[k].shape and k.endswith("out.0.weight"):
            st[k] = torch.cat([v, v.new_zeros(v.shape[0], cur[k].shape[1] - v.shape[1])], 1)
    missing, unexpected = m.load_state_dict(st, strict=False)
    allowed = ("mom_scale", "eq_scale", "inv_scale", "calibrated", "pqs_gate.weight", "tmix3.weight")
    assert not unexpected and all(k.endswith(allowed) for k in missing), (missing, unexpected)
    return m.eval()


def evaluate(m, ds, idx, atoms, bs=32):
    dE, q, qref, Q, Z, which = [], [], [], [], [], []
    for k in range(0, len(idx), bs):
        sel = idx[k:k + bs]
        g = ds.batch(sel)
        o = m(g)
        n = torch.zeros(int(g["batch"].max()) + 1, device="cuda").index_add(0, g["batch"], torch.ones_like(g["batch"], dtype=torch.float32))
        dE.append(((o["energy"] - g["E"]) / n).detach().cpu().numpy())
        q.append(g["q"].detach().cpu().numpy())
        for s in sel:
            at = atoms[s]
            qref.append(at.arrays["ref_q"]); Z.append(at.numbers)
            Q.append(np.full(len(at), at.info.get("charge", 0.0)))
            which.append(np.arange(len(at)))
    return (np.concatenate(dE), np.concatenate(q), np.concatenate(qref), np.concatenate(Q), np.concatenate(Z),
            np.concatenate(which))


if __name__ == "__main__":
    safety.install()
    torch.set_default_dtype(torch.float32)
    out = {}
    for key, tag in RUNS.items():
        ck = torch.load(f"runs/{tag}.pt", map_location="cuda", weights_only=False)
        ref = json.load(open(f"runs/{tag}.json"))["test"]["E_MAE_meV_atom"]
        a = ck["args"]
        atoms = read_runner(a["data"])
        perm = np.random.default_rng(a["seed"]).permutation(len(atoms))
        te = perm[a["n_train"] + a["n_val"]:a["n_train"] + a["n_val"] + a["n_test"]]
        ds = GraphSet(atoms, a["rc"], E0=ck["E0"])
        best = None
        for unit in (True, False):
            try:
                r = evaluate(build(ck, unit), ds, te, atoms)
            except RuntimeError:
                continue
            mae = 1e3 * np.abs(r[0]).mean()
            if best is None or abs(mae - ref) < abs(best[0] - ref):
                best = (mae, unit, r)
        mae, unit, (dE, q, qref, Q, Z, _) = best
        r_p = np.corrcoef(q, qref)[0, 1]
        print(f"{key}: unit={unit} E MAE {mae:.3f} (stored {ref:.3f}) meV/atom | charges vs Hirshfeld: "
              f"Pearson {r_p:.3f}, MAE {np.abs(q - qref).mean():.3f} e, mean |q| {np.abs(q).mean():.3f} / ref {np.abs(qref).mean():.3f}")
        out[f"{key}_q"], out[f"{key}_qref"], out[f"{key}_Q"], out[f"{key}_Z"] = q, qref, Q, Z
    np.savez_compressed("runs/paper_ko_charges.npz", **out)
