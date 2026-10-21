"""Localise training divergence: constant lr Adam, report first non-finite step per configuration."""
import sys, numpy as np, torch
sys.path.insert(0, ".")
from face import safety
from face.data import load_rmd17, MoleculeSet
from face.model import FACE, unit_parameterise
safety.install()
MOL = sys.argv[3] if len(sys.argv) > 3 else "aspirin"
NSTEP = int(sys.argv[4]) if len(sys.argv) > 4 else 400
Z, R, E, Fo, idx = load_rmd17(MOL)
ds = MoleculeSet(Z, R, E - E[idx["train"]].mean(), Fo, "cuda", torch.float32)
lr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
CFG = {
    "full v6 (calib)":            dict(kw=dict(transport=True, eqmsg=True), max_body=5),
    "max_body 3":                 dict(kw=dict(transport=True, eqmsg=True), max_body=3),
    "max_body 4":                 dict(kw=dict(transport=True, eqmsg=True), max_body=4),
    "no transport, no eqmsg":     dict(kw=dict(transport=False, eqmsg=False), max_body=5),
    "1 pass":                     dict(kw=dict(n_passes=1), max_body=5),
    "1 pass, max_body 3":         dict(kw=dict(n_passes=1), max_body=3),
    "full v6 + unit param":       dict(kw=dict(transport=True, eqmsg=True), max_body=5, unit=True),
    "v4 (LN, no calib) + unit":   dict(kw=dict(transport=True, eqmsg=True), max_body=5, unit=True, calib=False),
    "v4 (LN, no calib)":          dict(kw=dict(transport=True, eqmsg=True), max_body=5, calib=False),
    "v4 + unit + calib(LN kept)": dict(kw=dict(transport=True, eqmsg=True), max_body=5, unit=True, calib_ln=True),
}
sel = sys.argv[2].split(";") if len(sys.argv) > 2 else list(CFG)
for name in sel:
    c = CFG[name]
    torch.manual_seed(0); rng = np.random.default_rng(0)
    m = FACE(sorted(set(Z.tolist())), C=96, factor=True, B=32, K=32, **c["kw"]).cuda().set_calib(c.get("calib", True), keep_ln=c.get("calib_ln", False)).set_max_body(c["max_body"])
    if c.get("unit"):
        unit_parameterise(m)
    m.scale.fill_(float(ds.F[idx['train']].std())); m.avg_nb.fill_(14.4 if MOL == 'aspirin' else 8.0)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, amsgrad=True)
    first_bad, Fh = None, []
    for step in range(NSTEP):
        g = ds.batch(rng.choice(idx["train"], 5))
        o = m(g, create_graph=True)
        lF = ((o["forces"] - g["F"]) ** 2).mean()
        loss = (((o["energy"] - g["E"]) / ds.N) ** 2).mean() + 1000 * lF
        opt.zero_grad(); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 10.0)
        if not (torch.isfinite(loss) and torch.isfinite(gn)):
            first_bad = step; break
        opt.step()
        if step % (NSTEP // 5) == NSTEP // 5 - 1: Fh.append(round(1e3 * lF.sqrt().item(), 1))
    print(f"{MOL} lr={lr} {name:26s} first non-finite step: {first_bad}   train F-RMSE every 100 steps: {Fh}", flush=True)
