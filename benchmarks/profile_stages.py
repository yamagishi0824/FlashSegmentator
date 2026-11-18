"""Fine-grained stage profile of the ORIGINAL `total` path (warm models), per case.

python benchmarks/profile_stages.py <data_dir> <weights> <out_dir> [n_cases]

Wraps the functions the original path calls (in the namespaces where they are looked up) with
cuda-synchronised timers. Nested calls are reported inclusively; `forward` is the sliding window.
"""
import collections
import glob
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import SimpleITK as sitk
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "flashseg"))
import batched as B  # noqa: E402
ti = B.ti
import nnunet_infer_nii as nn_nii  # noqa: E402

acc = collections.defaultdict(float)


def T():
    torch.cuda.synchronize(); return time.perf_counter()


def wrap(owner, attr, label):
    f = getattr(owner, attr)

    def w(*a, **k):
        t = T(); r = f(*a, **k); acc[label] += T() - t; return r
    setattr(owner, attr, w)


torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
files = sorted(glob.glob(sys.argv[1] + "/*.nii.gz"))[:int(sys.argv[4]) if len(sys.argv) > 4 else 10]
cfg = ti.TASK_CONFIGS["total"]; dev = torch.device("cuda")
preds = ti.build_predictors(cfg, Path(sys.argv[2]), dev, 0.8)
out = Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)

# --- instrument (names as looked up by totalseg_infer / nnunet_infer_nii) ---
wrap(nib, "load", "nib.load")
wrap(nib, "as_closest_canonical", "as_closest_canonical")
wrap(ti, "tseg_change_spacing", "change_spacing(cucim fwd+inv)")
wrap(nib.Nifti1Image, "get_fdata", "get_fdata (float64)")
wrap(ti, "_merge_part", "merge_part (numpy)")
wrap(ti, "remove_auxiliary_labels", "remove_auxiliary_labels")
wrap(ti, "tseg_undo_canonical", "undo_canonical")
wrap(sitk, "WriteImage", "sitk.WriteImage (gzip)")
wrap(sitk, "ReadImage", "sitk.ReadImage (CopyInformation)")
wrap(nn_nii, "convert_predicted_logits_to_segmentation_with_correct_shape", "logits->seg (resample+threshold)")
for p in preds.values():
    wrap(p, "preprocess", "preprocess (crop+norm+resample)")
    wrap(p, "_internal_predict_sliding_window_return_logits", "forward (sliding window)")

ti.infer_file(files[0], preds, cfg, "total", str(out), cache={})  # warm-up
acc.clear()
t0 = T()
for f in files:
    ti.infer_file(f, preds, cfg, "total", str(out), cache={})
total = T() - t0
n = len(files)
print(f"{n} cases, {total / n:.2f} s/case")
rows = sorted(acc.items(), key=lambda kv: -kv[1])
for k, v in rows:
    print(f"{v / n:7.3f} s/case  {100 * v / total:5.1f}%  {k}")
json.dump({"n": n, "s_per_case": total / n, "stages_s_per_case": {k: v / n for k, v in rows}},
          open(out / "profile.json", "w"), indent=1)
