"""Write rMD17 train/val/test splits (identical to face.data) as extxyz for MACE."""
import sys, os
import numpy as np
from ase import Atoms
from ase.io import write
from ase.calculators.singlepoint import SinglePointCalculator
sys.path.insert(0, ".")
from face.data import load_rmd17

mol = sys.argv[1] if len(sys.argv) > 1 else "aspirin"
Z, R, E, Fo, idx = load_rmd17(mol)
os.makedirs("data/xyz", exist_ok=True)
for split, ii in idx.items():
    frames = []
    for i in ii:
        a = Atoms(numbers=Z, positions=R[i])
        a.info["REF_energy"] = float(E[i])
        a.arrays["REF_forces"] = Fo[i]
        frames.append(a)
    write(f"data/xyz/{mol}_{split}.xyz", frames, format="extxyz")
    print(split, len(frames))
