"""FlashSegmentator CLI: TotalSegmentator `total` (5 part-models) over a folder of CTs, with the model loaded once,
pipelined prep/GPU/write, GPU logits->label, and static-shape torch.compile. See README / docs/PERF_LOG.md.

python flashseg/cli.py -i <ct_dir_or_file> -o <out_dir> [--batch 8] [--group 4] [--no_compile] [--weights DIR]
"""
import argparse
import glob
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batched as B  # noqa: E402
import static_compile  # noqa: E402

ti = B.ti


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", required=True, help="a .nii.gz file or a directory of them")
    ap.add_argument("-o", "--output", required=True, help="output directory (one multilabel .nii.gz per input)")
    ap.add_argument("--weights", default=str(Path.home() / ".totalsegmentator/nnunet/results"))
    ap.add_argument("--batch", type=int, default=8, help="patches per forward (GH200: 8, smaller GPUs: 2)")
    ap.add_argument("--group", type=int, default=4, help="cases pooled per pipeline stage")
    ap.add_argument("--no_compile", action="store_true", help="skip static torch.compile (e.g. T4 / quick runs)")
    a = ap.parse_args()
    files = [a.input] if os.path.isfile(a.input) else sorted(glob.glob(os.path.join(a.input, "*.nii.gz")))
    if not files:
        sys.exit(f"no .nii.gz found in {a.input}")
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    cfg, task, dev = ti.TASK_CONFIGS["total"], "total", torch.device("cuda")
    t = time.perf_counter()
    preds = ti.build_predictors(cfg, Path(a.weights), dev, ti._step_size_for(task, cfg))
    if not a.no_compile:
        static_compile.apply(preds, a.batch)
    print(f"{len(files)} cases, models loaded in {time.perf_counter() - t:.1f}s", flush=True)
    t = time.perf_counter()
    B.run_pipeline(files, preds, cfg, task, a.batch, a.group, out, fast_post=True,
                   on_done=lambda f: print("done", os.path.basename(str(f)), flush=True))
    dt = time.perf_counter() - t
    print(f"{len(files)} cases in {dt:.1f}s ({3600 * len(files) / dt:.0f} CT/h, incl. compile warm-up)", flush=True)


if __name__ == "__main__":
    main()
