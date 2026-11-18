"""Per-case end-to-end time of the ORIGINAL path, split into stages (warm models)."""
import glob, sys, time, json
from pathlib import Path
import torch, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "flashseg"))
import batched as B
ti = B.ti
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
files = sorted(glob.glob(sys.argv[1] + "/*.nii.gz"))[:12]
cfg = ti.TASK_CONFIGS["total"]; dev = torch.device("cuda")
preds = ti.build_predictors(cfg, Path(sys.argv[2]), dev, 0.8)
out = Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
def T(): torch.cuda.synchronize(); return time.perf_counter()
# instrument the forward (sliding window) of every predictor
fw = {"t": 0.0}
for p in preds.values():
    orig = p._internal_predict_sliding_window_return_logits
    def wrap(*a, _o=orig, **k):
        t = T(); r = _o(*a, **k); fw["t"] += T() - t; return r
    p._internal_predict_sliding_window_return_logits = wrap
ti.infer_file(files[0], preds, cfg, "total", str(out), cache={})  # warm-up
rows = []
for f in files:
    fw["t"] = 0.0; t = T()
    ti.infer_file(f, preds, cfg, "total", str(out), cache={})
    tot = T() - t
    rows.append(dict(case=Path(f).name, total_s=tot, forward_s=fw["t"], other_s=tot - fw["t"]))
    print(rows[-1], flush=True)
print("sum total %.1fs  forward %.1fs  other %.1fs  -> %.0f CT/h" % (
    sum(r["total_s"] for r in rows), sum(r["forward_s"] for r in rows),
    sum(r["other_s"] for r in rows), 12 / sum(r["total_s"] for r in rows) * 3600))
json.dump(rows, open(out / "percase.json", "w"), indent=1)
