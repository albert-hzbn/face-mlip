import sys, torch
sys.path.insert(0, ".")
from face import safety
from face.model import FACE
from face.neighbors import padded_graph
from tools.bench import aspirin_box
from torch.profiler import profile, ProfilerActivity
safety.install()
at = aspirin_box(6)
from face.model import unit_parameterise
m = unit_parameterise(FACE([1, 6, 8], C=96, transport=True, eqmsg=True, factor=True, B=16, cp=True).cuda().eval()).fuse()
g = padded_graph([at], m.rc)
def f():
    gg = dict(g); gg["pos"] = g["pos"].clone(); return m(gg)["forces"]
for _ in range(3): f()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    f(); torch.cuda.synchronize()
ev = prof.key_averages(group_by_input_shape=True)
rows = sorted(ev, key=lambda e: -e.self_device_time_total)[:25]
tot = sum(e.self_device_time_total for e in ev)
print(f"total CUDA {tot/1e3:.1f} ms, kernels {sum(e.count for e in ev)}")
for e in rows:
    print(f"{e.self_device_time_total/1e3:8.2f} ms  x{e.count:<4d} {e.key[:28]:28s} {str(e.input_shapes)[:110]}")
