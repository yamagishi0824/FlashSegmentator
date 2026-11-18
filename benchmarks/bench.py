"""Benchmark: original FastSegmentator (`total`) vs batched inference.

python benchmarks/bench.py --data ../data/btcv --n_tp 12 --group 4 --batches 1 2 4 8 16 32 --out ../bench_out

* model load time measured separately (predictors built once, shared by all conditions)
* latency  : 1 CT, end-to-end (read → write), warm-up 1 + `--reps` timed runs, median
* throughput: `--n_tp` CTs, batched path processes `--group` CTs per GPU group; CT/hour
* memory   : torch.cuda.max_memory_allocated peak per condition (unified-memory GPUs: includes host-side use)
* agreement: per-case mean Dice over labels present in either, vs the original output
"""
import argparse
import glob
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "flashseg"))
import batched as B  # noqa: E402
ti = B.ti


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def dice_mean(a, b):
    labs = np.union1d(np.unique(a), np.unique(b)); labs = labs[labs > 0]
    d = []
    for l in labs:
        x, y = a == l, b == l
        d.append(2 * np.logical_and(x, y).sum() / (x.sum() + y.sum()))
    return float(np.mean(d)) if d else 1.0, float(np.min(d)) if d else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n_tp", type=int, default=12); ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--backend", default="eager", choices=["eager", "trt_fp16", "trt_fp32"])
    ap.add_argument("--ref_dir", default=None, help="dir with eager-original outputs (default: <out>/original)")
    ap.add_argument("--weights", default=str(Path.home() / ".totalsegmentator/nnunet/results"))
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    a = ap.parse_args()
    # identical determinism settings to totalseg_infer.main()
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    files = sorted(glob.glob(os.path.join(a.data, "*.nii.gz")))[:a.n_tp]
    lat_file = files[0]
    cfg, task = ti.TASK_CONFIGS["total"], "total"
    dev = torch.device("cuda")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res = {"files": [Path(f).name for f in files], "gpu": torch.cuda.get_device_name(),
           "torch": torch.__version__, "conditions": {}}

    t0 = sync_time()
    preds = ti.build_predictors(cfg, Path(a.weights), dev,
                                ti._step_size_for(task, cfg))
    res["model_load_s"] = sync_time() - t0
    print(f"model load {res['model_load_s']:.1f}s", flush=True)
    res["backend"] = a.backend
    if a.backend.startswith("trt"):
        # network -> TensorRT engine (cached on disk); same call site in original & batched paths
        from trt_backend import get_trt_module
        res["trt_timing"] = {}
        for tid, p in preds.items():
            mod, tm = get_trt_module(p.network.eval(), p.list_of_parameters[-1],
                                     (1, *p.configuration_manager.patch_size), a.backend[4:])
            p.network = mod
            res["trt_timing"][tid] = tm
            print(tid, tm, flush=True)

    def run_orig(fs, od):
        for f in fs:
            ti.infer_file(f, preds, cfg, task, str(od), weights_dir=None, device=dev, cache={})

    def run_batched(fs, od, bs, group):
        st = {'bs': bs, 'oom_events': 0}
        for i in range(0, len(fs), group):
            B.run_group(fs[i:i + group], preds, cfg, task, bs, out_dir=od, state=st)
        return st

    conds = [("original", None)] + [(f"batched_b{b}", b) for b in a.batches]
    for name, bs in conds:
        od = out / name; od.mkdir(exist_ok=True)
        r = {}
        try:
            # latency (single CT); warm-up once
            lat = []
            for k in range(a.reps + 1):
                torch.cuda.reset_peak_memory_stats()
                t = sync_time()
                run_orig([lat_file], od) if bs is None else run_batched([lat_file], od, bs, 1)
                if k: lat.append(sync_time() - t)
            r["latency_s_median"] = statistics.median(lat); r["latency_s_all"] = lat
            r["latency_peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
            # throughput
            torch.cuda.reset_peak_memory_stats()
            t = sync_time()
            st = run_orig(files, od) if bs is None else run_batched(files, od, bs, a.group)
            el = sync_time() - t
            r.update(throughput_wall_s=el, ct_per_hour=len(files) / el * 3600,
                     tp_peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30,
                     group=None if bs is None else a.group)
            if st: r["oom_events"] = st['oom_events']; r["final_bs"] = st['bs']
        except torch.OutOfMemoryError as e:
            r["error"] = f"OOM: {e}"[:200]
        torch.cuda.empty_cache()
        res["conditions"][name] = r
        print(name, json.dumps(r), flush=True)
        json.dump(res, open(out / "results.json", "w"), indent=1)

    # agreement vs original
    import nibabel as nib
    for name, _ in (conds if a.ref_dir else conds[1:]):
        ds = []
        for f in files:
            n = Path(f).name.replace("_0000.nii.gz", "").replace(".nii.gz", "") + ".nii.gz"
            p = out / name / n
            if p.exists():
                ref = np.asarray(nib.load(Path(a.ref_dir or out / "original") / n).dataobj)
                ds.append(dice_mean(ref, np.asarray(nib.load(p).dataobj)) +
                          (float((ref == np.asarray(nib.load(p).dataobj)).mean()),
                           int((ref != np.asarray(nib.load(p).dataobj)).sum())))
        if ds:
            res["conditions"][name]["dice_mean"] = float(np.mean([d[0] for d in ds]))
            res["conditions"][name]["dice_min_label"] = float(np.min([d[1] for d in ds]))
            res["conditions"][name]["voxel_agreement"] = float(np.mean([d[2] for d in ds]))
            res["conditions"][name]["diff_voxels_per_case_mean"] = float(np.mean([d[3] for d in ds]))
            res["conditions"][name]["diff_voxels_per_case_max"] = int(np.max([d[3] for d in ds]))
        print(name, {k: res["conditions"][name].get(k) for k in ("dice_mean", "dice_min_label", "voxel_agreement")}, flush=True)
    json.dump(res, open(out / "results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
