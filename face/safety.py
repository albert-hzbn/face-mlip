"""Resource guards so long jobs cannot take the workstation down.

- caps this process's CUDA allocations (prevents the Windows WDDM driver from spilling
  into shared system RAM, which freezes the desktop) -> a clean OutOfMemoryError instead
- lowers process priority and CPU thread count so the desktop stays responsive
- watchdog thread: exits cleanly if free system RAM or GPU temperature cross limits
"""
import os, sys, threading, time, subprocess
import psutil
import torch

GPU_FRACTION = float(os.environ.get("FACE_GPU_FRACTION", 0.75))
MIN_FREE_RAM_GB = float(os.environ.get("FACE_MIN_FREE_RAM_GB", 8))
MAX_GPU_TEMP_C = float(os.environ.get("FACE_MAX_GPU_TEMP", 87))
CPU_THREADS = int(os.environ.get("FACE_CPU_THREADS", 8))


def _gpu_temp():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout
        return float(out.strip().splitlines()[0])
    except Exception:
        return None


def _watchdog(period=5.0):
    hot = 0
    while True:
        free_gb = psutil.virtual_memory().available / 2**30
        if free_gb < MIN_FREE_RAM_GB:
            print(f"[safety] free RAM {free_gb:.1f} GB < {MIN_FREE_RAM_GB} GB -> aborting", flush=True)
            os._exit(3)
        t = _gpu_temp()
        hot = hot + 1 if (t is not None and t >= MAX_GPU_TEMP_C) else 0
        if hot >= 3:
            print(f"[safety] GPU at {t:.0f} C for 15 s -> aborting", flush=True)
            os._exit(4)
        time.sleep(period)


def install():
    p = psutil.Process()
    try:
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if sys.platform == "win32" else 10)
    except Exception:
        pass
    torch.set_num_threads(CPU_THREADS)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(GPU_FRACTION)
    threading.Thread(target=_watchdog, daemon=True).start()
