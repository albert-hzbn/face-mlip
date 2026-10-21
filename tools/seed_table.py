"""Mean and standard deviation over three seeds for the equal-budget (60-epoch) comparison of Table 5.

FACE flagship: seed 0 from the original suite (runs/suite_flag_<mol>.json, aspirin: runs/v7_B16_rc45_aspirin.json),
seeds 1-2 from runs/seedsuite_flag_s<k>_<mol>.json. MACE-small: seeds 0-2 with final-epoch evaluation
(runs/seedsuite_mace_s<k>_<mol>_eval.txt). Ethanol FACE: the three seeds of the original study (flag_ethanol,
flag_s1, s2_rec). Writes runs/seed_table.json.
"""
import json, os, re
import numpy as np

SEED0 = {"aspirin": "runs/v7_B16_rc45_aspirin.json"}
ETH = ["runs/flag_ethanol.json", "runs/flag_s1_ethanol.json", "runs/s2_rec_ethanol.json"]


def face(mol):
    files = ETH if mol == "ethanol" else [SEED0.get(mol, f"runs/suite_flag_{mol}.json")] + \
        [f"runs/seedsuite_flag_s{k}_{mol}.json" for k in (1, 2)]
    out = []
    for f in files:
        if os.path.exists(f):
            d = json.load(open(f)); out.append((d["test_E_meV"], d["test_F_meV_A"]))
    return out


def mace(mol):
    out = []
    for k in range(3):
        f = f"runs/seedsuite_mace_s{k}_{mol}_eval.txt"
        if os.path.exists(f) and os.path.getsize(f):
            d = json.loads(re.search(r"\{.*\}", open(f).read()).group(0)); out.append((d["E_MAE"], d["F_MAE"]))
    return out


res = {}
for mol in ("aspirin", "ethanol", "malonaldehyde", "uracil", "toluene"):
    r = {}
    for name, vals in (("FACE", face(mol)), ("MACE", mace(mol))):
        if vals:
            a = np.array(vals)
            r[name] = {"n": len(vals), "E_mean": a[:, 0].mean(), "E_std": a[:, 0].std(ddof=1) if len(vals) > 1 else 0.0,
                       "F_mean": a[:, 1].mean(), "F_std": a[:, 1].std(ddof=1) if len(vals) > 1 else 0.0, "values": vals}
    res[mol] = r
    line = "  ".join(f"{n} (n={v['n']}): {v['E_mean']:.2f}±{v['E_std']:.2f} / {v['F_mean']:.2f}±{v['F_std']:.2f}" for n, v in r.items())
    print(f"{mol:14s} {line}")
json.dump(res, open("runs/seed_table.json", "w"), indent=1)
