"""Patch micro-benchmark of cheap knobs on one model: cudnn.benchmark / deterministic, channels_last_3d,
eager vs torch.compile(static), and 5-model fused vs separate. Each knob combo in-process (cudnn flags
are global, set per row). python benchmarks/micro_knobs.py <weights> <out.json>"""
import sys, time, json, itertools
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "flashseg"))
import batched as B, fused
ti = B.ti
preds = ti.build_predictors(ti.TASK_CONFIGS['total'], Path(sys.argv[1]), torch.device('cuda'), 0.8)
nets = [p.network.eval() for p in preds.values()]
res = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "rows": []}

def tm(fn, x, n=8):
    for _ in range(3): fn(x)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn(x)
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e3

for bench, cl, comp in itertools.product([False, True], [False, True], [False, True]):
    torch.backends.cudnn.benchmark = bench; torch.backends.cudnn.deterministic = not bench
    torch._dynamo.reset()
    mf = torch.channels_last_3d if cl else torch.contiguous_format
    ns = [n.to(memory_format=mf) for n in nets]
    fns = [torch.compile(n, dynamic=False) if comp else n for n in ns]
    for Bsz in (1, 2, 4):
        x = torch.randn(Bsz, 1, 128, 128, 128, device='cuda').contiguous(memory_format=mf)
        def run(x):
            with torch.no_grad(), torch.autocast('cuda'):
                return [f(x) for f in fns]
        ms = tm(run, x)
        row = dict(cudnn_benchmark=bench, channels_last=cl, compile=comp, B=Bsz, ms_5models=ms, ms_per_patch_per_model=ms / Bsz / 5)
        res["rows"].append(row); print(row, flush=True)
    json.dump(res, open(sys.argv[2], "w"), indent=1)
# fused vs separate (eager & compile, best cudnn setting = benchmark on)
torch.backends.cudnn.benchmark = True; torch.backends.cudnn.deterministic = False
nets = [n.to(memory_format=torch.contiguous_format) for n in nets]
for comp in (False, True):
    torch._dynamo.reset()
    f = fused.FusedUNet(nets).cuda().eval()
    fn = torch.compile(f, dynamic=False) if comp else f
    for Bsz in (1, 2):
        x = torch.randn(Bsz, 5, 128, 128, 128, device='cuda')
        def run(x):
            with torch.no_grad(), torch.autocast('cuda'):
                return fn(x)
        ms = tm(run, x)
        row = dict(fused=True, cudnn_benchmark=True, compile=comp, B=Bsz, ms_5models=ms, ms_per_patch_per_model=ms / Bsz / 5)
        res["rows"].append(row); print(row, flush=True)
json.dump(res, open(sys.argv[2], "w"), indent=1)
