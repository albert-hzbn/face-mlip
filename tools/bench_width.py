"""Cost and memory versus model width (FACE node width C, MACE channels) and memory versus system size.

usage: python tools/bench_width.py width   -> runs/bench_width.json   (1,344 atoms, interleaved timing rounds)
       python tools/bench_width.py memory  -> runs/bench_memory.json  (peak GPU memory, energy + forces)
Timing is load-robust (models interleaved in rounds, medians); every time is also reported relative to MACE-small
measured in the same rounds.
"""
import sys, time, json, statistics, gc
import torch
sys.path.insert(0, ".")
from face import safety
from tools.bench import aspirin_box
from tools.bench_interleaved import face_fn, mace_fn

FLAG = dict(transport=True, eqmsg=True, factor=True, K=32, B=16, rc=4.5)
CPK = dict(transport=True, eqmsg=True, factor=True, K=32, B=16, cp=True)


def width():
    at = aspirin_box(4)
    models = {f"MACE {c}": (lambda c=c: mace_fn(at, channels=c)) for c in (32, 64, 128)}
    models.update({f"FACE C{c}": (lambda c=c: face_fn(at, True, True, C=c, **FLAG)) for c in (32, 64, 96, 128, 192, 256)})
    models.update({f"FACE-CP C{c}": (lambda c=c: face_fn(at, True, False, C=c, **CPK)) for c in (64, 96, 160)})
    fns = {k: make() for k, make in models.items()}
    times = {k: [] for k in fns}
    for f in fns.values():
        for _ in range(3):
            f()
    for _ in range(7):
        for k, f in fns.items():
            torch.cuda.synchronize(); t = time.perf_counter()
            for _ in range(5):
                f()
            torch.cuda.synchronize(); times[k].append((time.perf_counter() - t) / 5)
    ref = statistics.median(times["MACE 64"])
    out = {"atoms": len(at)}
    for k, v in times.items():
        med = statistics.median(v)
        out[k] = {"us_per_atom_loaded": 1e6 * med / len(at), "time_rel_MACE64": med / ref}
        print(f"{k:14s} {1e6 * med / len(at):8.1f} us/atom (loaded GPU)   time / MACE-64 = {med / ref:5.3f}", flush=True)
    json.dump(out, open("runs/bench_width.json", "w"), indent=1)


def peak(make):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    try:
        f = make(); f(); torch.cuda.synchronize()
        p = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
    except RuntimeError as e:          # OutOfMemoryError, or a RuntimeError from TorchScript'ed MACE code
        if "out of memory" not in str(e):
            raise
        p = None
    f = None; gc.collect(); torch.cuda.empty_cache()
    return p


def memory():
    out = {}
    for side in (2, 3, 4, 5, 6):
        at = aspirin_box(side)
        row = {"MACE 64": peak(lambda: mace_fn(at, channels=64)),
               "MACE 128": peak(lambda: mace_fn(at, channels=128)),
               "FACE flagship": peak(lambda: face_fn(at, True, True, C=96, **FLAG)),
               "FACE-CP C96": peak(lambda: face_fn(at, True, False, C=96, **CPK))}
        out[len(at)] = row
        print(len(at), {k: (None if v is None else round(v)) for k, v in row.items()}, flush=True)
    json.dump(out, open("runs/bench_memory.json", "w"), indent=1)


if __name__ == "__main__":
    safety.install()
    torch.set_default_dtype(torch.float32)
    {"width": width, "memory": memory}[sys.argv[1]]()
