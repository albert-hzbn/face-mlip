import sys, torch
sys.path.insert(0, ".")
from face import safety
from face.model import FACE
from face.neighbors import padded_graph
from tools.bench import aspirin_box
from torch.profiler import profile, ProfilerActivity

safety.install()
torch.set_default_dtype(torch.float32)
at = aspirin_box(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
m = FACE([1, 6, 8], C=96, transport=True, eqmsg=True, factor=True).cuda().eval()
g = padded_graph([at], 5.0)
print("atoms", len(at), "M", g["nbr"].shape[1], "valid nbrs/atom", g["mask"].sum(1).float().mean().item())


def f():
    gg = dict(g); gg["pos"] = g["pos"].clone()
    return m(gg)["forces"]


for _ in range(5):
    f()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
    for _ in range(5):
        f()
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
