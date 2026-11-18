"""Benchmark v2 (many cases): original vs batched vs batched+pipeline, `total` task.

python benchmarks/bench2.py --data <dir of *.nii.gz> --weights <results> --out <dir> \
    --batches 1 2 4 --pipe_batches 1 2 4 --group 4 [--limit N]

latency   : each case processed alone (group=1, no pipeline), wall time per case (read→write);
            reported as median / p10 / p90 / max over cases. One warm-up case first.
throughput: all cases in one pass (batched: groups of --group; pipeline: overlapped), CT/hour.
agreement : Dice / differing voxels vs the original outputs (written to <out>/original).
memory    : torch.cuda.max_memory_allocated during the throughput pass.
"""
import argparse, glob, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "flashseg"))
import batched as B  # noqa: E402
from bench import dice_mean  # noqa: E402
ti = B.ti


def T():
    torch.cuda.synchronize(); return time.perf_counter()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--batches", type=int, nargs="*", default=[1, 2, 4])
    ap.add_argument("--pipe_batches", type=int, nargs="*", default=[1, 2, 4])
    ap.add_argument("--group", type=int, default=4); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no_latency", action="store_true")
    ap.add_argument("--no_original", action="store_true")
    ap.add_argument("--static_compile", default=None,
                    help="torch.compile mode for static-shape compile of pipefast conditions (PERF_LOG 6)")
    ap.add_argument("--fast_post_batches", type=int, nargs="*", default=[], help="pipeline + fastpost (PERF_LOG 5a/5b)")
    ap.add_argument("--cudnn_benchmark", action="store_true", help="PERF_LOG §13: cudnn.benchmark for the fast paths (shorter compile warm-up)")
    ap.add_argument("--no_dice", action="store_true", help="skip agreement (compute offline)")
    ap.add_argument("--ref_dir", default=None, help="reference outputs (default <out>/original)")
    a = ap.parse_args()
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    files = sorted(glob.glob(os.path.join(a.data, "*.nii.gz")))[:a.limit]
    cfg, task, dev = ti.TASK_CONFIGS["total"], "total", torch.device("cuda")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res = {"n_cases": len(files), "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
           "group": a.group, "conditions": {}}
    t = T(); preds = ti.build_predictors(cfg, Path(a.weights), dev, ti._step_size_for(task, cfg))
    res["model_load_s"] = T() - t
    print(f"{len(files)} cases, model load {res['model_load_s']:.1f}s", flush=True)

    def one_orig(f, od): ti.infer_file(f, preds, cfg, task, str(od), cache={})
    def one_batched(f, od, bs): B.run_group([f], preds, cfg, task, bs, out_dir=od)

    conds = ([] if a.no_original else [("original", None, False)]) + [(f"batched_b{b}", b, False) for b in a.batches] \
        + [(f"pipeline_b{b}", b, True) for b in a.pipe_batches] \
        + [(f"pipefast{'_sc' if a.static_compile else ''}_b{b}", b, "fast") for b in a.fast_post_batches]
    (out / "_warm").mkdir(exist_ok=True)
    res["nnUNet_compile"] = os.environ.get("nnUNet_compile", "")
    def warm(): (B.run_pipeline(files[:1], preds, cfg, task, 1, 1, out / "_warm", fast_post=True)
                 if a.no_original else one_orig(files[0], out / "_warm"))
    t = T(); warm(); res["warmup_first_case_s"] = T() - t   # incl. torch.compile
    t = T(); warm(); res["warm_second_case_s"] = T() - t
    print("warm-up", res["warmup_first_case_s"], res["warm_second_case_s"], flush=True)
    base_nets = {tid: p.network for tid, p in preds.items()}
    for name, bs, pipe in conds:
        od = out / name; od.mkdir(exist_ok=True); r = {}
        if a.cudnn_benchmark and pipe == "fast":
            torch.backends.cudnn.deterministic = False; torch.backends.cudnn.benchmark = True
        if a.static_compile and pipe == "fast":
            import static_compile
            for tid, p in preds.items():
                p.network = static_compile.StaticBatchCompiled(base_nets[tid].eval(), bs, a.static_compile)
            t = T(); B.run_pipeline(files[:1], preds, cfg, task, bs, 1, out / "_warm", fast_post=True)
            r["compile_warmup_s"] = T() - t
            if not a.no_latency:   # per-case latency of the fast path (one case at a time)
                lat = []
                for f in files:
                    t = T(); B.run_pipeline([f], preds, cfg, task, bs, 1, od, fast_post=True); lat.append(T() - t)
                q = np.percentile(lat, [10, 50, 90])
                r.update(latency_median_s=float(q[1]), latency_p10_s=float(q[0]), latency_p90_s=float(q[2]),
                         latency_max_s=float(max(lat)), latency_mean_s=float(np.mean(lat)))

        failed = []
        if not a.no_latency and not pipe:
            lat = []
            for f in files:
                t = T()
                try:
                    one_orig(f, od) if bs is None else one_batched(f, od, bs)
                    lat.append(T() - t)
                except Exception as e:
                    failed.append(Path(f).name); print("FAILED", f, e, flush=True); torch.cuda.empty_cache()
            q = np.percentile(lat, [10, 50, 90])
            r.update(latency_median_s=float(q[1]), latency_p10_s=float(q[0]), latency_p90_s=float(q[2]),
                     latency_max_s=float(max(lat)), latency_mean_s=float(np.mean(lat)), latency_failed=failed)
        torch.cuda.reset_peak_memory_stats(); st = {'bs': bs or 1, 'oom_events': 0}
        t = T()
        if bs is None:
            for f in files:
                try: one_orig(f, od)
                except Exception as e: print("FAILED", f, e, flush=True); torch.cuda.empty_cache()
        elif not pipe:
            for i in range(0, len(files), a.group):
                try: B.run_group(files[i:i + a.group], preds, cfg, task, bs, out_dir=od, state=st)
                except Exception as e: print("FAILED group", i, e, flush=True); torch.cuda.empty_cache()
        else:
            B.run_pipeline(files, preds, cfg, task, bs, a.group, od, state=st, fast_post=(pipe == "fast"))
        el = T() - t
        r.update(throughput_s=el, ct_per_hour=len(files) / el * 3600, mean_s_per_ct=el / len(files),
                 peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30, oom_events=st['oom_events'],
                 wait_prep_s=st.get('wait_prep_s'), wait_write_tail_s=st.get('wait_write_tail_s'),
                 **{k: st[k] for k in ('t_stack_s', 't_fwd_s', 't_accum_s', 'n_patches', 'n_calls') if k in st})
        res["conditions"][name] = r
        print(name, json.dumps(r), flush=True)
        json.dump(res, open(out / "results.json", "w"), indent=1)
        torch.cuda.empty_cache()

    if a.no_dice:
        return
    import nibabel as nib
    ref = Path(a.ref_dir) if a.ref_dir else out / "original"
    for name, _, _ in (conds if (a.ref_dir or a.no_original) else conds[1:]):
        ds = []
        for f in files:
            n = Path(f).name.replace("_0000.nii.gz", "").replace(".nii.gz", "") + ".nii.gz"
            p, q = out / name / n, ref / n
            if p.exists() and q.exists():
                x, y = np.asarray(nib.load(q).dataobj), np.asarray(nib.load(p).dataobj)
                dm, dmin = dice_mean(x, y); ds.append((dm, dmin, int((x != y).sum()), x.size))
        c = res["conditions"][name]
        c.update(dice_mean=float(np.mean([d[0] for d in ds])), dice_min_label=float(np.min([d[1] for d in ds])),
                 diff_voxels_total=int(sum(d[2] for d in ds)), diff_voxel_frac=float(sum(d[2] for d in ds) / sum(d[3] for d in ds)),
                 cases_identical=int(sum(d[2] == 0 for d in ds)), cases_compared=len(ds))
        print(name, {k: c[k] for k in ("dice_mean", "dice_min_label", "diff_voxels_total", "cases_identical", "cases_compared")}, flush=True)
    json.dump(res, open(out / "results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
