"""Why is batch >=4 slower? (PERF_LOG §12) One fresh process per (mode, B):
python benchmarks/probe_batch.py <weights> <out.jsonl> <mode> <B> [--prof]
mode: eager_det | eager_bench | compile_det | compile_bench
Records ms/patch (5 models), peak memory, dynamo recompiles/graph breaks; --prof dumps top CUDA kernels."""
import sys, time, json
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "flashseg"))
import batched as B
ti = B.ti
w, out, mode, Bsz = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
prof = "--prof" in sys.argv
bench = "bench" in mode
if "lay" in mode:
    import torch._inductor.config as ic; ic.force_layout_optimization = True
mf = torch.channels_last_3d if "_cl" in mode else torch.contiguous_format
torch.backends.cudnn.benchmark = bench; torch.backends.cudnn.deterministic = not bench
preds = ti.build_predictors(ti.TASK_CONFIGS['total'], Path(w), torch.device('cuda'), 0.8)
nets = [p.network.eval().to(memory_format=mf) for p in preds.values()]
fns = [torch.compile(n, dynamic=False) for n in nets] if mode.startswith("compile") else nets
x = torch.randn(Bsz, 1, 128, 128, 128, device='cuda').contiguous(memory_format=mf)
def run():
    with torch.no_grad(), torch.autocast('cuda'):
        return [f(x) for f in fns]
t0 = time.perf_counter(); run(); torch.cuda.synchronize(); first = time.perf_counter() - t0
for _ in range(3): run()
torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
ts = []
for _ in range(10):
    torch.cuda.synchronize(); t = time.perf_counter(); run(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
ts.sort(); ms = ts[len(ts) // 2] * 1e3
row = dict(mode=mode, B=Bsz, ms_5models=ms, ms_per_patch_per_model=ms / Bsz / 5, first_call_s=first,
           peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30, gpu=torch.cuda.get_device_name(), torch=torch.__version__)
try:
    from torch._dynamo.utils import counters
    row["dynamo"] = {k: dict(v) for k, v in counters.items() if k in ("stats", "graph_break", "recompiles")}
except Exception as e:
    row["dynamo"] = str(e)
if prof:
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        run(); torch.cuda.synchronize()
    ev = sorted(p.key_averages(), key=lambda e: -e.device_time_total)[:20]
    row["top_kernels"] = [dict(name=e.key[:120], ms=e.device_time_total / 1e3, n=e.count) for e in ev]
print(json.dumps(row)[:600], flush=True)
open(out, "a").write(json.dumps(row) + "\n")
