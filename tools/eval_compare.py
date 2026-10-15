"""Same-metric comparison on the identical rMD17 test split: MAE and RMSE of total energy
(meV) and force components (meV/A), for a FACE checkpoint and a MACE model."""
import sys, json
import numpy as np
import torch
sys.path.insert(0, ".")
from face import safety
from face.data import load_rmd17, MoleculeSet
from face.model import FACE, unit_parameterise


def stats(dE, dF):
    return {"E_MAE": 1e3 * np.abs(dE).mean(), "E_RMSE": 1e3 * np.sqrt((dE ** 2).mean()),
            "F_MAE": 1e3 * np.abs(dF).mean(), "F_RMSE": 1e3 * np.sqrt((dF ** 2).mean())}


def eval_face(ckpt, mol):
    c = torch.load(ckpt, weights_only=False)
    a = c["args"]
    torch.set_default_dtype(getattr(torch, a["dtype"]))
    Z, R, E, Fo, idx = load_rmd17(mol)
    terms = [t for t in a["terms"].split(",") if t]
    m = FACE(c["elements"], C=a["C"], n_basis=a["nbasis"], rc=a["rc"], n_passes=a["passes"], terms=terms,
               radial_hidden=a.get("radial_hidden", 64), K_dir=a.get("kdir", 16), norm=a.get("norm", "layer"),
               transport=bool(a.get("transport", 0)), eqmsg=bool(a.get("eqmsg", 0)),
               factor=bool(a.get("factor", 0)), B=a.get("B", 16), K=a.get("K", 32), cp=bool(a.get("cp", 0))).cuda()
    m.set_standardise(bool(a.get("standardise", 0)))
    m.set_momnorm(bool(a.get("momnorm", 0)))
    m.set_calib(bool(a.get("calib", 0)), keep_ln=bool(a.get("calib_ln", 0)))
    m.set_eq_linear(bool(a.get("eq_linear", 0)))
    m.set_max_body(a.get("max_body", 5))
    m.set_elem_proj(bool(a.get("elem_proj", 0)))
    m.set_poly_readout(bool(a.get("poly", 0)))
    m.set_transport3(bool(a.get("t3", 0)))
    if a.get("unit", 0):
        unit_parameterise(m)
    missing, unexpected = m.load_state_dict(c["state"], strict=False)
    assert not unexpected and all(k.endswith(("inv_scale", "calibrated", "mom_scale", "eq_scale", "pqs_gate.weight", "tmix3.weight")) for k in missing), (missing, unexpected)
    m.eval()
    ds = MoleculeSet(Z, R, E - c["Eref"], Fo, "cuda", torch.get_default_dtype())
    dE, dF = [], []
    te = idx["test"]
    for k in range(0, len(te), 50):
        g = ds.batch(te[k:k + 50]); o = m(g)
        dE.append((o["energy"] - g["E"]).detach().double().cpu().numpy())
        dF.append((o["forces"] - g["F"]).detach().double().cpu().numpy().ravel())
    return stats(np.concatenate(dE), np.concatenate(dF))


def eval_mace(model_path, mol):
    from mace.calculators import MACECalculator
    from ase import Atoms
    calc = MACECalculator(model_paths=model_path, device="cuda", default_dtype="float32")
    Z, R, E, Fo, idx = load_rmd17(mol)
    dE, dF = [], []
    for i in idx["test"]:
        at = Atoms(numbers=Z, positions=R[i]); at.calc = calc
        dE.append(at.get_potential_energy() - E[i]); dF.append((at.get_forces() - Fo[i]).ravel())
    return stats(np.array(dE), np.concatenate(dF))


if __name__ == "__main__":
    safety.install()
    mol = sys.argv[1]
    out = {}
    for spec in sys.argv[2:]:
        kind, path = spec.split("=", 1)
        out[path] = eval_face(path, mol) if kind == "face" else eval_mace(path, mol)
        print(path, json.dumps({k: round(v, 3) for k, v in out[path].items()}), flush=True)
