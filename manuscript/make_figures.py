"""Figures for the manuscript. Run from the project root:  ../.venv/Scripts/python manuscript/make_figures.py
All numbers are measured values from runs/ and docs/FACE.md (sections given in comments)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "manuscript/figures"
os.makedirs(OUT, exist_ok=True)
# Okabe-Ito colour-blind-safe palette (validated: lightness, chroma, CVD separation, contrast), used sparingly:
# black/grey for structure, blue as the main accent, vermillion and green only where a second or third series is needed.
BLUE, ORANGE, AQUA, YELLOW = "#0072B2", "#D55E00", "#009E73", "#222222"
INK, INK2, MUTED, GRID = "#000000", "#262626", "#7a7a7a", "#d9d9d9"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11, "legend.fontsize": 9.5,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.edgecolor": INK, "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "axes.spines.top": True, "axes.spines.right": True, "axes.grid": False,
    "xtick.color": INK, "ytick.color": INK, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "xtick.major.size": 3, "ytick.major.size": 3,
    "lines.linewidth": 1.1, "lines.markersize": 4.5, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight"})


def label(ax, x, y, s, dx=4, dy=3, ha="left", color=INK2):
    ax.annotate(s, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=10, color=color, ha=ha)


# ---------------------------------------------------------------- Fig. 2: accuracy vs cost (aspirin)
fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 3.9))
# (a) equal 60-epoch budget, idle-GPU timings at 1,344 atoms (docs §5, §4.12; CP time from interleaved ratio §4.25)
mace = [("MACE-small", 190.9, 19.5), ("MACE-medium", 361.5, 17.8)]
mos = [("FACE flagship", 33.3, 18.3), ("FACE C64, $r_c$ 4.5", 30.4, 21.5), ("FACE-CP", 66.0, 16.4)]
a.plot([m[1] for m in mace], [m[2] for m in mace], "s", color=ORANGE, ms=7, mec=INK, mew=0.5, label="MACE (trained here)")
a.plot([m[1] for m in mos], [m[2] for m in mos], "o", color=BLUE, ms=7, mec=INK, mew=0.5, label="FACE")
for n, x, y in mace:                       # labels beside the markers, vertically centred, clear of everything else
    a.annotate(n, (x, y), xytext=(-7, 0), textcoords="offset points", fontsize=10, color=INK2, ha="right", va="center")
for n, x, y in mos:
    a.annotate(n, (x, y), xytext=(7, 0), textcoords="offset points", fontsize=10, color=INK2, ha="left", va="center")
a.set_xscale("log"); a.set_xlim(20, 600); a.set_ylim(15.8, 24.2); a.set_yticks(range(16, 23))
a.set_xticks([25, 50, 100, 200, 400]); a.set_xticklabels(["25", "50", "100", "200", "400"]); a.minorticks_off()
a.set_xlabel("cost (µs per atom and step)"); a.set_ylabel("aspirin force MAE (meV/Å)")
a.set_title("(a) equal budget, 60 epochs", fontsize=11, loc="left", color=INK)
a.legend(frameon=False, fontsize=9.5, loc="upper right", handletextpad=0.3)
# (b) best single models after long training / distillation (docs §4.13, §4.23–4.30) vs published long-training values
singles = [("flagship, distilled", 33.3, 8.04), ("CP $C$=96, gen 3", 66.0, 6.97), ("CP $C$=160, gen 4", 97.0, 6.74)]
# distinct markers + legend instead of point labels (no text near lines or markers)
b.plot([s[1] for s in singles], [s[2] for s in singles], "-", color=BLUE, lw=1.0, zorder=2)
for (n, x, y), mk in zip(singles, ["o", "s", "^"]):
    b.plot([x], [y], mk, color=BLUE, ms=7, mec=INK, mew=0.5, zorder=3, label=n)
b.plot([1634], [6.6], "D", color=MUTED, ms=6, mec=INK, mew=0.5, label="MACE (published)")
# published reference lines are named in the right margin, outside the plotting area
for name, y in [("GotenNet", 5.66), ("Allegro", 7.3), ("NequIP", 8.2), ("TensorNet", 8.9)]:
    b.axhline(y, color=MUTED, lw=0.6, ls=(0, (4, 2)), zorder=0)
    b.text(1.02, y, name, transform=b.get_yaxis_transform(), fontsize=9.5, color=MUTED, ha="left", va="center")
b.set_xscale("log"); b.set_xlim(20, 3000); b.set_ylim(5.2, 10.6)
b.set_xticks([30, 100, 300, 1000, 3000]); b.set_xticklabels(["30", "100", "300", "1000", "3000"]); b.minorticks_off()
b.set_xlabel("cost (µs per atom and step)"); b.set_ylabel("aspirin force MAE (meV/Å)")
b.set_title("(b) best single models", fontsize=11, loc="left", color=INK)
leg = b.legend(fontsize=9, loc="upper right", labelspacing=0.25, handletextpad=0.3, borderpad=0.2,
               handlelength=1.0, frameon=False)   # above the TensorNet line, clear of all data
fig.tight_layout()
fig.savefig(f"{OUT}/fig_cost_accuracy.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig. 4: distillation
fig = plt.figure(figsize=(6.5, 7.0))
gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1])
a = fig.add_subplot(gs[0, :]); b = fig.add_subplot(gs[1, 0]); c = fig.add_subplot(gs[1, 1])
# (a) teacher force spread (docs §4.23): random rattles vs teacher MD vs real test set
# single-line labels set diagonally so that the long descriptions stay readable
lab = ["rattle σ = 0.02–0.03 Å", "rattle σ = 0.03–0.04 Å", "rattle σ = 0.04–0.05 Å", "rattle σ = 0.05–0.06 Å",
       "teacher MD, 500 K", "test set"]
val = [9.5, 22.5, 60.7, 150.1, 6.9, 5.5]
col = ["#bdbdbd"] * 4 + [BLUE, "#555555"]
bars = a.bar(range(len(val)), val, color=col, width=0.7, edgecolor=INK, linewidth=0.5)
for i, v in enumerate(val):
    a.text(i, v * 1.12, f"{v:g}", ha="center", fontsize=9.5, color=INK2)
a.set_yscale("log"); a.set_ylim(3, 400)
a.set_xticks(range(len(val)))
a.set_xticklabels(lab, fontsize=10, rotation=25, ha="right", rotation_mode="anchor")
a.set_ylabel("teacher force spread (meV/Å)")
a.set_title("(a) where the teacher is reliable", fontsize=11, loc="left")
a.grid(axis="x", visible=False)
# (b, c) generations (docs §4.13, §4.18, §4.23–4.30); separate panels -> no dual axis
gen = ["DFT only", "gen 1", "gen 2", "gen 3", "gen 4, C160", "gen 4, E-fine-tune"]
F = [8.73, 7.46, 7.04, 6.97, 6.70, 6.74]
E = [2.88, 2.71, 2.69, 2.50, 2.48, 2.26]
for ax, y, ref, name in [(b, F, 6.6, "force MAE (meV/Å)"), (c, E, 2.2, "energy MAE (meV)")]:
    ax.plot(range(len(y)), y, "o-", color=BLUE, ms=6, mec=INK, mew=0.5, label="FACE-CP")
    ax.set_ylim(ref - 0.08 * (max(y) - ref), max(y) + 0.08 * (max(y) - ref))
    ax.axhline(ref, color=MUTED, lw=0.8, ls=(0, (4, 2)), label="MACE (published)")
    ax.legend(loc="upper right", fontsize=9.5, handlelength=1.6)
    ax.set_xticks(range(len(y))); ax.set_xticklabels(gen, fontsize=10, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylabel(name)
b.set_title("(b) forces, single FACE-CP model", fontsize=11, loc="left")
c.set_title("(c) energies, single FACE-CP model", fontsize=11, loc="left")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_distillation.pdf")
plt.close(fig)


# ---------------------------------------------------------------- Fig. 3: learning curves (validation, from logs)
def curve(path):
    ep, vf = [], []
    for line in open(path, encoding="utf-8", errors="ignore"):
        if line.startswith("{"):
            d = json.loads(line)
            ep.append(d["epoch"]); vf.append(d["val_F"])
    return np.array(ep), np.array(vf)


fig, ax = plt.subplots(figsize=(4.2, 3.2))
series = [("runs/long_flag_aspirin.log", "flagship (C96 B16)", BLUE, "o"),
          ("runs/long_cp_aspirin.log", "CP (C96)", AQUA, "s"),
          ("runs/long_L_aspirin.log", "FACE-L (C192)", YELLOW, "^"),
          ("runs/long_v7_aspirin.log", "B32 K$_d$32", ORANGE, "D")]
for path, name, col, mk in series:
    e, v = curve(path)
    ax.plot(e, v, "-", color=col, marker=mk, ms=3.5, markevery=4, label=name)
ax.set_yscale("log"); ax.set_ylim(7, 120)
ax.set_xlabel("epoch"); ax.set_ylabel("validation force MAE (meV/Å)")
ax.legend(frameon=False, fontsize=9.5)
ax.set_title("aspirin, 400 epochs, lr 0.04, unit parameterisation", fontsize=10, loc="left")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_learning_curves.pdf")
plt.close(fig)
print("figures written to", OUT)

# ---------------------------------------------------------------- TOC graphic (ACS: 3.25 x 1.75 in)
fig, ax = plt.subplots(figsize=(3.25, 1.75))
names = ["MACE-medium", "MACE-small", "FACE"]
cost = [361.5, 190.9, 33.3]          # µs/atom/step, idle GPU, 1,344 atoms (Table 2)
fmae = [17.8, 19.5, 18.3]            # aspirin force MAE, equal 60-epoch budget (meV/Å)
cols = ["#bdbdbd", "#bdbdbd", BLUE]
ax.barh(range(3), cost, color=cols, height=0.62, edgecolor=INK, linewidth=0.5)
for i, (c, f) in enumerate(zip(cost, fmae)):
    ax.text(c + 8, i, f"{c:.0f} µs/atom ({f} meV/Å)", va="center", fontsize=7, color=INK2)
ax.set_yticks(range(3)); ax.set_yticklabels(names, fontsize=9)
ax.set_xlim(0, 820); ax.set_xlabel("µs per atom and step (energy + forces)", fontsize=8)
ax.grid(axis="y", visible=False); ax.tick_params(axis="x", labelsize=8)
ax.set_title("Inference cost at matched accuracy (aspirin)", fontsize=8.5, loc="left", color=INK)
fig.tight_layout(pad=0.3)
with plt.rc_context({"savefig.bbox": "standard"}):   # exact ACS size 3.25 x 1.75 in (no tight cropping)
    fig.savefig(f"{OUT}/toc.pdf")
plt.close(fig)
print("toc written")

# ---------------------------------------------------------------- Fig. 6: error analysis (runs/paper_preds_aspirin.npz,
# produced by tools/paper_data.py preds: best single model = gen-4 FACE-CP, ensemble = five flagship models)
from scipy.stats import spearmanr
ELEM = [(1, "H", BLUE, "-"), (6, "C", ORANGE, "--"), (8, "O", AQUA, ":")]
d = np.load("runs/paper_preds_aspirin.npz")
Z, Fr, Fb, Fe = d["Z"], d["F_ref"], d["F_best"], d["F_ens"]
fig, (a, b, c) = plt.subplots(1, 3, figsize=(6.5, 3.35))
hb = a.hexbin(Fr.ravel(), Fb.ravel(), gridsize=60, bins="log", cmap="Greys", mincnt=1, linewidths=0)
lim = 9.5
a.plot([-lim, lim], [-lim, lim], color=MUTED, lw=0.6, ls=(0, (4, 2)))
a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
a.set_xlabel("DFT force (eV/Å)"); a.set_ylabel("FACE force (eV/Å)")
a.set_title("(a) parity", fontsize=11, loc="left", color=INK)
a.text(0.04, 0.96, "MAE 6.74 meV/Å\nRMSE 10.4 meV/Å", transform=a.transAxes, va="top", fontsize=9.5, color=INK2)
a.set_xticks([-5, 0, 5]); a.set_yticks([-5, 0, 5])
for z, n, col, ls in ELEM:
    e = np.sort(np.abs(Fb[:, Z == z] - Fr[:, Z == z]).ravel()) * 1e3
    b.plot(e, np.arange(1, len(e) + 1) / len(e), color=col, ls=ls, label=n)   # per-element MAEs are given in the text
b.set_xscale("log"); b.set_xlim(0.05, 200); b.set_ylim(0, 1.0)
b.set_xlabel("|force error| (meV/Å)"); b.set_ylabel("cumulative fraction")
b.set_title("(b) by element", fontsize=11, loc="left", color=INK)
b.legend(frameon=False, fontsize=9.5, loc="center left")
sp = Fe.std(0).mean((1, 2)) * 1e3
er = np.abs(Fe.mean(0) - Fr).mean((1, 2)) * 1e3
c.plot(sp, er, "o", ms=2.4, color="none", mec=BLUE, mew=0.5, alpha=0.7)
c.plot([2, 40], [2, 40], color=MUTED, lw=0.6, ls=(0, (4, 2)))
c.set_xscale("log"); c.set_yscale("log"); c.set_xlim(2, 40); c.set_ylim(2, 40)
c.set_xticks([2, 5, 10, 20]); c.set_xticklabels(["2", "5", "10", "20"])
c.set_yticks([2, 5, 10, 20]); c.set_yticklabels(["2", "5", "10", "20"]); c.minorticks_off()
c.set_xlabel("force spread (meV/Å)"); c.set_ylabel("force MAE (meV/Å)")
c.set_title("(c) uncertainty", fontsize=11, loc="left", color=INK)
fig.tight_layout(w_pad=1.2)
fig.savefig(f"{OUT}/fig_errors.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig. 5: speed-up vs system size (runs/bench_scaling.txt,
# interleaved medians vs MACE-small; n^3 aspirin molecules, 21 atoms each)
atoms = [168, 567, 1344, 2625]
flag = [2.34, 4.67, 5.64, 5.43]
cpv = [2.37, 2.77, 3.20, 2.91]
fig, ax = plt.subplots(figsize=(4.2, 3.0))
ax.plot(atoms, flag, "o-", color=BLUE, ms=6, mec=INK, mew=0.5, label="FACE flagship")
ax.plot(atoms, cpv, "s--", color=ORANGE, ms=6, mec=INK, mew=0.5, label="FACE-CP")
ax.axhline(1.0, color=MUTED, lw=0.8, ls=(0, (4, 2)), label="MACE-small (reference)")
ax.legend(loc="upper left", fontsize=9.5)
ax.set_xscale("log"); ax.set_xticks(atoms); ax.set_xticklabels([str(n) for n in atoms]); ax.minorticks_off()
ax.set_ylim(0, 7.5); ax.set_xlabel("atoms"); ax.set_ylabel("speed-up vs MACE-small")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_scaling.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig. 7: molecular dynamics (tools/paper_data.py md / langevin)
if os.path.exists("runs/paper_nve_aspirin.npz") and os.path.exists("runs/paper_langevin_aspirin.npz"):
    import sys; sys.path.insert(0, ".")
    from face.data import load_rmd17
    nve = np.load("runs/paper_nve_aspirin.npz"); lgv = np.load("runs/paper_langevin_aspirin.npz")
    n = len(nve["Z"]); t = nve["t_ps"]
    Et = (nve["Epot"] + nve["Ekin"]) / n * 1e3; Ep = nve["Epot"] / n * 1e3
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 3.20))
    a.plot(t, Ep - Ep.mean(), color="#a0a0a0", lw=0.4, label="potential energy (shifted)")
    a.plot(t, Et - Et[0], color=BLUE, lw=1.2, label="total energy")
    a.set_xlabel("time (ps)"); a.set_ylabel("energy (meV/atom)"); a.set_xlim(0, t[-1])
    a.set_title("(a) NVE, 500 K start, 0.5 fs step", fontsize=11, loc="left", color=INK)
    a.set_ylim(-32, 62); a.legend(frameon=False, fontsize=9.5, loc="upper left")
    iu = np.triu_indices(n, 1)
    P = lgv["pos"][lgv["t_ps"] > 1.0]          # discard the first ps (equilibration)
    dmd = np.linalg.norm(P[:, :, None] - P[:, None], axis=-1)[:, iu[0], iu[1]].ravel()
    Zr, Rr, *_ = load_rmd17("aspirin")
    sel = np.random.default_rng(0).choice(len(Rr), 5000, replace=False)
    dref = np.linalg.norm(Rr[sel][:, :, None] - Rr[sel][:, None], axis=-1)[:, iu[0], iu[1]].ravel()
    bins = np.linspace(0.8, 7.5, 271)
    hr, _ = np.histogram(dref, bins, density=True); hm, _ = np.histogram(dmd, bins, density=True)
    ctr = 0.5 * (bins[1:] + bins[:-1])
    b.fill_between(ctr, hr, color="#c8c8c8", lw=0, label="rMD17 (ab initio MD)")
    b.plot(ctr, hm, color=BLUE, lw=1.0, label="FACE (Langevin MD)")
    b.set_xlim(0.8, 7.5); b.set_ylim(0, 0.78)
    b.set_xlabel("interatomic distance (Å)"); b.set_ylabel("probability density (1/Å)")
    b.set_title("(b) distance distribution", fontsize=11, loc="left", color=INK)
    b.legend(frameon=False, fontsize=9.5, loc="upper right")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(f"{OUT}/fig_md.pdf")
    plt.close(fig)
    print("md figure written")
# ---------------------------------------------------------------- Fig.: cost vs width and memory vs size
# (tools/bench_width.py width / memory -> runs/bench_width.json, runs/bench_memory.json)
bw = json.load(open("runs/bench_width.json")); bm = json.load(open("runs/bench_memory.json"))
fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 3.35))
fam = [("MACE", "MACE", [32, 64, 128], ORANGE, "s", "--"),
       ("FACE C", "FACE flagship", [32, 64, 96, 128, 192, 256], BLUE, "o", "-"),
       ("FACE-CP C", "FACE-CP", [64, 96, 160], AQUA, "^", ":")]
for key, name, ws, col, mk, ls in fam:
    k = (lambda w: f"MACE {w}") if key == "MACE" else (lambda w, key=key: f"{key}{w}")
    a.plot(ws, [bw[k(w)]["time_rel_MACE64"] for w in ws], marker=mk, ls=ls, color=col, ms=5.5, mec=INK, mew=0.5,
           label=name)
a.set_xscale("log", base=2); a.set_xticks([32, 64, 128, 256]); a.set_xticklabels(["32", "64", "128", "256"])
a.minorticks_off(); a.set_ylim(0, 2.1)
a.set_xlabel("channels (FACE: node width $C$)"); a.set_ylabel("time relative to MACE, 64 channels")
a.set_title("(a) cost versus width, 1,344 atoms", fontsize=11, loc="left", color=INK)
a.legend(loc="upper left", fontsize=9.5)
sizes = sorted(int(n) for n in bm)
for name, lab, col, mk, ls in [("MACE 128", "MACE, 128 channels", ORANGE, "D", "--"),
                               ("MACE 64", "MACE, 64 channels", ORANGE, "s", "-"),
                               ("FACE-CP C96", "FACE-CP", AQUA, "^", ":"),
                               ("FACE flagship", "FACE flagship", BLUE, "o", "-")]:
    pts = [(n, bm[str(n)][name] / 1024) for n in sizes if bm[str(n)][name] is not None]
    b.plot(*zip(*pts), marker=mk, ls=ls, color=col, ms=5.5, mec=INK, mew=0.5, label=lab)
b.axhline(12, color=MUTED, lw=0.8, ls=(0, (4, 2)), label="12 GB memory cap")
b.set_xscale("log"); b.set_yscale("log"); b.set_xticks(sizes); b.set_xticklabels([str(n) for n in sizes]); b.minorticks_off()
b.set_yticks([0.1, 1, 10, 100]); b.set_yticklabels(["0.1", "1", "10", "100"]); b.set_ylim(0.08, 600)
b.set_xlabel("atoms"); b.set_ylabel("peak GPU memory (GB), E+F")
b.set_title("(b) memory versus system size", fontsize=11, loc="left", color=INK)
b.legend(loc="upper left", fontsize=9)
fig.tight_layout(w_pad=1.5)
fig.savefig(f"{OUT}/fig_width_memory.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig.: latent charges vs reference Hirshfeld charges
# (tools/ko_charges.py -> runs/paper_ko_charges.npz; the models were never trained on charges)
kc = np.load("runs/paper_ko_charges.npz")
fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 3.55))
q, qr, Q = kc["Ag3_q"], kc["Ag3_qref"], kc["Ag3_Q"]
for tot, col, mk, lab in [(-1.0, BLUE, "o", "Ag$_3^-$"), (1.0, ORANGE, "s", "Ag$_3^+$")]:
    m = Q == tot
    a.plot(qr[m], q[m], mk, ms=2.5, color="none", mec=col, mew=0.5, alpha=0.7,
           label=f"{lab}, $r$ = {np.corrcoef(q[m], qr[m])[0, 1]:.2f}")
lo, hi = -0.75, 0.75
a.plot([lo, hi], [lo, hi], color=MUTED, lw=0.6, ls=(0, (4, 2)))
a.set_xlim(lo, hi); a.set_ylim(lo, hi); a.set_aspect("equal")
a.set_xlabel("Hirshfeld charge, DFT ($e$)"); a.set_ylabel("latent charge, FACE ($e$)")
a.set_title("(a) Ag$_3^\\pm$, coulomb mode", fontsize=11, loc="left", color=INK)
a.legend(loc="upper left", fontsize=9.5, markerscale=2)
q, qr, Q = kc["C10H2_rc3_q"], kc["C10H2_rc3_qref"], kc["C10H2_rc3_Q"]
for tot, col, mk, lab in [(0.0, BLUE, "o", "C$_{10}$H$_2$"), (1.0, ORANGE, "s", "C$_{10}$H$_3^+$")]:
    m = Q == tot
    b.plot(qr[m], q[m], mk, ms=2.0, color="none", mec=col, mew=0.4, alpha=0.5,
           label=f"{lab}, $r$ = {np.corrcoef(q[m], qr[m])[0, 1]:.2f}")
lo, hi = min(qr.min(), q.min()) - 0.02, max(qr.max(), q.max()) + 0.02
b.plot([lo, hi], [lo, hi], color=MUTED, lw=0.6, ls=(0, (4, 2)))
b.set_xlim(lo, hi); b.set_ylim(lo, hi); b.set_aspect("equal")
b.set_xlabel("Hirshfeld charge, DFT ($e$)"); b.set_ylabel("latent charge, FACE ($e$)")
b.set_title("(b) C$_{10}$H$_2$/C$_{10}$H$_3^+$, qeq mode, $r_c$ = 3 Å", fontsize=11, loc="left", color=INK)
b.legend(loc="lower right", fontsize=9.5, markerscale=2)
fig.tight_layout(w_pad=2.0)
fig.savefig(f"{OUT}/fig_charges.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig.: learning curve (runs/lc_n*_aspirin.json)
lc = [(n, f"runs/lc_n{n}_aspirin.json") for n in (50, 100, 250, 500)] + [(950, "runs/long_flag_aspirin.json")]
if all(os.path.exists(p) for _, p in lc):
    res = [(n, json.load(open(p))) for n, p in lc]
    N = [n for n, _ in res]
    Ef = [d["test_E_meV"] for _, d in res]; Ff = [d["test_F_meV_A"] for _, d in res]
    # published rMD17 aspirin, MACE paper Table 1 (N = 50 and N = 1000/950)
    pub = {"MACE": ([50, 950], [17.0, 2.2], [43.9, 6.6]), "NequIP": ([50, 950], [19.5, 2.3], [52.0, 8.2]),
           "ACE": ([50, 950], [26.2, 6.1], [63.8, 17.9])}
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 3.35))
    for ax, y, idx, lab in [(a, Ff, 2, "force MAE (meV/Å)"), (b, Ef, 1, "energy MAE (meV)")]:
        ax.plot(N, y, "o-", color=BLUE, ms=5.5, mec=INK, mew=0.5, label="FACE flagship (this work)")
        for (name, v), (col, mk) in zip(pub.items(), [(ORANGE, "s"), (AQUA, "^"), (YELLOW, "D")]):
            ax.plot(v[0], v[idx], marker=mk, ls="none", color=col, ms=6.5, mec=INK, mew=0.5, label=f"{name} (published)")   # two published points only: no connecting line
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks([50, 100, 250, 500, 950])
        ax.set_xticklabels(["50", "100", "250", "500", "950"]); ax.minorticks_off()
        ax.set_xlabel("training structures"); ax.set_ylabel(lab)
    a.set_title("(a) forces", fontsize=11, loc="left", color=INK); b.set_title("(b) energies", fontsize=11, loc="left", color=INK)
    h, l = a.get_legend_handles_labels()       # one shared legend below both panels, clear of all data
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=9.5, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(w_pad=1.5, rect=(0, 0.17, 1, 1))
    fig.savefig(f"{OUT}/fig_learning_curve.pdf")
    plt.close(fig)
    print("learning-curve figure written")
print("extra figures written")
