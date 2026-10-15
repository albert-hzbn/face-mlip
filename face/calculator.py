"""ASE calculator for FACE checkpoints (molecules or periodic cells)."""
import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes

from .model import FACE, unit_parameterise
from .datasets import GraphSet


def load_face(path, device="cuda"):
    c = torch.load(path, weights_only=False, map_location=device)
    a = c["args"]
    terms = [t for t in a.get("terms", "").split(",") if t]
    m = FACE(c.get("elements", sorted(c.get("E0", {}))), C=a["C"], n_basis=a.get("nbasis", 8), rc=a["rc"],
               n_passes=a["passes"], terms=terms, radial_hidden=a.get("radial_hidden", 64),
               K_dir=a.get("kdir", 16), norm=a.get("norm", "layer"), transport=bool(a.get("transport", 0)),
               eqmsg=bool(a.get("eqmsg", 0)), factor=bool(a.get("factor", 0)), B=a.get("B", 16),
               K=a.get("K", 32), cp=bool(a.get("cp", 0))).to(device)
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
    return m, c


class FACECalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, path, device="cuda", fused=True, e_shift=0.0, **kw):
        super().__init__(**kw)
        self.model, ck = load_face(path, device)
        self.model.fuse(fused and device == "cuda")
        if fused and device == "cuda" and getattr(self.model.blocks[0], "spline", 1) is None:
            self.model.compile_radial()
        self.device = device
        self.e_shift = e_shift + float(ck.get("Eref", 0.0))

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        at = self.atoms.copy()
        at.info["energy"] = 0.0
        at.arrays["forces"] = np.zeros((len(at), 3))
        g = GraphSet([at], self.model.rc, device=self.device, dtype=torch.float32).batch([0])
        o = self.model(g)
        self.results["energy"] = float(o["energy"].item()) + self.e_shift
        self.results["forces"] = o["forces"].detach().double().cpu().numpy()


class FACEEnsembleCalculator(Calculator):
    """Deep ensemble of FACE checkpoints: mean energy/forces; member spread as uncertainty
    (results['energy_std'] in eV, results['forces_std'] per atom in eV/A). On aspirin the per-structure force spread
    rank-correlates with the true error at 0.82-0.84 (docs/FACE.md 4.20)."""
    implemented_properties = ["energy", "forces"]

    def __init__(self, paths, device="cuda", fused=True, **kw):
        super().__init__(**kw)
        self.members = [FACECalculator(p, device=device, fused=fused) for p in paths]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        Es, Fs = [], []
        for c in self.members:
            c.calculate(self.atoms, properties, system_changes)
            Es.append(c.results["energy"]); Fs.append(c.results["forces"])
        Es, Fs = np.array(Es), np.array(Fs)
        self.results["energy"] = float(Es.mean())
        self.results["forces"] = Fs.mean(0)
        self.results["energy_std"] = float(Es.std())
        self.results["forces_std"] = np.linalg.norm(Fs.std(0), axis=-1)
