"""FlashSegmentator CLI: TotalSegmentator `total` (5 part-models) over a folder of CTs, with the model loaded once,
pipelined prep/GPU/write, GPU logits->label, and static-shape torch.compile. See README / docs/PERF_LOG.md.

python flashseg/cli.py -i <ct_dir_or_file> -o <out_dir> [--batch 8] [--group 4] [--no_compile] [--weights DIR] [--log [DIR]]

With --log, a run log is written to <out_dir>/flashseg_log/ (or DIR): summary.json (throughput excl. warm-up, model load,
first-case/compile time, peak GPU memory, GPU, versions, settings) and per_case.csv (completion time of every CT).
"""
import argparse
import glob
import os
import sys
import json
import platform
import subprocess
import time
from datetime import datetime
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
    ap.add_argument("--log", nargs="?", const="", default=None, metavar="DIR",
                    help="write a run log (default dir <out_dir>/flashseg_log)")
    a = ap.parse_args()
    files = [a.input] if os.path.isfile(a.input) else sorted(glob.glob(os.path.join(a.input, "*.nii.gz")))
    if not files:
        sys.exit(f"no .nii.gz found in {a.input}")
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    cfg, task, dev = ti.TASK_CONFIGS["total"], "total", torch.device("cuda")
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    T = time.perf_counter
    t = T()
    preds = ti.build_predictors(cfg, Path(a.weights), dev, ti._step_size_for(task, cfg))
    if not a.no_compile:
        static_compile.apply(preds, a.batch)
    model_load_s = T() - t
    print(f"{len(files)} cases, models loaded in {model_load_s:.1f}s", flush=True)
    torch.cuda.reset_peak_memory_stats()
    rows = []
    t_run = T()

    def done(f):
        rows.append((os.path.basename(str(f)), T() - t_run))
        print("done", rows[-1][0], f"{rows[-1][1]:.1f}s", flush=True)

    # first CT alone: includes the one-off torch.compile warm-up, reported separately from throughput
    B.run_pipeline(files[:1], preds, cfg, task, a.batch, 1, out, fast_post=True, on_done=done)
    first_case_s = T() - t_run
    rest = files[1:]
    t = T()
    if rest:
        B.run_pipeline(rest, preds, cfg, task, a.batch, a.group, out, fast_post=True, on_done=done)
    rest_s = T() - t
    total_s = T() - t_run
    thr = 3600 * len(rest) / rest_s if rest else None
    msg = f"{len(files)} cases in {total_s:.1f}s; first case (incl. compile) {first_case_s:.1f}s"
    if thr:
        msg += f"; throughput {thr:.0f} CT/h ({rest_s / len(rest):.2f} s/CT) over the remaining {len(rest)}"
    print(msg, flush=True)
    if a.log is None:
        return
    log = Path(a.log) if a.log else out / "flashseg_log"
    log.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        commit = None
    summary = dict(
        started=started, n_cases=len(files), input=str(a.input), output=str(out),
        settings=dict(batch=a.batch, group=a.group, compile=not a.no_compile, cudnn_benchmark=True, task=task),
        model_load_s=round(model_load_s, 3), first_case_incl_compile_s=round(first_case_s, 3),
        throughput_ct_per_hour=round(thr, 1) if thr else None,
        mean_s_per_ct_excl_first=round(rest_s / len(rest), 3) if rest else None,
        total_wall_s=round(total_s, 3), peak_gpu_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
        python=platform.python_version(), flashsegmentator_commit=commit)
    json.dump(summary, open(log / "summary.json", "w"), indent=1)
    with open(log / "per_case.csv", "w") as fo:
        fo.write("case,done_at_s,interval_s\n")
        prev = 0.0
        for n, s_ in rows:
            fo.write(f"{n},{s_:.3f},{s_ - prev:.3f}\n"); prev = s_
    print(f"log: {log}/summary.json, per_case.csv", flush=True)

if __name__ == "__main__":
    main()
