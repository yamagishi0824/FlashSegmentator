"""Paired equivalence of Dice (TOST) between segmentation outputs and a reference method (docs/PERF_LOG.md §16).

python benchmarks/equivalence.py --gt <gt_dir> --ref name=dir --pred name=dir [name=dir ...] --out eq.json
       [--label_map benchmarks/btcv_label_map.json] [--delta 0.01]

For every (case, label) present in the GT, Dice of each method vs GT. Then, for each method vs --ref:
  * case level : per-case mean Dice, paired difference, 90% CI (t), TOST p-value for |diff| < delta
  * label level: per-label paired difference over the cases containing that label (n >= 3),
                 90% CI and TOST; number of labels declared equivalent at delta
TOST with 90% CI == two one-sided tests at alpha 0.05.
"""
import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import stats

from eval_gt import case_dice


def tost(d, delta):
    d = np.asarray(d, float); n = d.size
    m = d.mean(); sd = d.std(ddof=1) if n > 1 else 0.0
    if sd == 0:
        return dict(n=n, mean=m, ci90=[m, m], p_tost=0.0 if abs(m) < delta else 1.0, equivalent=bool(abs(m) < delta))
    se = sd / np.sqrt(n); t = stats.t.ppf(0.95, n - 1)
    p_lo = 1 - stats.t.cdf((m + delta) / se, n - 1)     # H0: diff <= -delta
    p_hi = stats.t.cdf((m - delta) / se, n - 1)         # H0: diff >= +delta
    p = float(max(p_lo, p_hi))
    return dict(n=n, mean=float(m), sd=float(sd), ci90=[float(m - t * se), float(m + t * se)], p_tost=p, equivalent=bool(p < 0.05))


def load(d, gt, lut, workers):
    cases = sorted(os.path.basename(f) for f in glob.glob(os.path.join(d, "*.nii.gz")))
    cases = [c for c in cases if os.path.exists(os.path.join(gt, c))]
    with ProcessPoolExecutor(workers) as ex:
        out = list(ex.map(case_dice, [(os.path.join(gt, c), os.path.join(d, c), lut) for c in cases]))
    return {c: o for c, o in zip(cases, out) if o}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True); ap.add_argument("--ref", required=True)
    ap.add_argument("--pred", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--label_map", default=None); ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    lut = None
    if a.label_map:
        lut = np.zeros(256, dtype=np.uint8)
        for k, v in json.load(open(a.label_map))["pred_to_gt"].items():
            lut[int(k)] = int(v)
    rname, rdir = a.ref.split("=", 1)
    ref = load(rdir, a.gt, lut, a.workers)
    res = {"ref": rname, "delta": a.delta, "n_ref_cases": len(ref), "methods": {}}
    for p in a.pred:
        name, d = p.split("=", 1)
        x = load(d, a.gt, lut, a.workers)
        common = sorted(set(x) & set(ref))
        cm_x = [np.mean(list(x[c].values())) for c in common]
        cm_r = [np.mean([ref[c][l] for l in x[c] if l in ref[c]]) for c in common]
        cm_x = [np.mean([x[c][l] for l in x[c] if l in ref[c]]) for c in common]
        case = tost(np.array(cm_x) - np.array(cm_r), a.delta)
        case.update(dice_method=float(np.mean(cm_x)), dice_ref=float(np.mean(cm_r)))
        labs = sorted({l for c in common for l in x[c] if l in ref[c]})
        per = {}
        for l in labs:
            dd = [x[c][l] - ref[c][l] for c in common if l in x[c] and l in ref[c]]
            if len(dd) >= 3:
                per[int(l)] = tost(dd, a.delta)
        allp = [x[c][l] - ref[c][l] for c in common for l in x[c] if l in ref[c]]
        ci_w = max(max(abs(v["ci90"][0]), abs(v["ci90"][1])) for v in per.values()) if per else None
        res["methods"][name] = {
            "n_cases": len(common), "case_level": case,
            "pairs": {"n": len(allp), "mean": float(np.mean(allp)), "abs_max": float(np.max(np.abs(allp))),
                      "p99_abs": float(np.percentile(np.abs(allp), 99))},
            "labels_tested": len(per), "labels_equivalent": int(sum(v["equivalent"] for v in per.values())),
            "max_abs_ci90_bound_over_labels": ci_w,
            "worst_labels": sorted(((k, v["ci90"]) for k, v in per.items()),
                                   key=lambda kv: -max(abs(kv[1][0]), abs(kv[1][1])))[:5],
            "per_label": per}
        m = res["methods"][name]
        print(f"{name} vs {rname}: cases {m['n_cases']}, case-mean diff {case['mean']:+.2e} CI90 [{case['ci90'][0]:+.2e},{case['ci90'][1]:+.2e}] "
              f"p_TOST {case['p_tost']:.1e}; labels equivalent {m['labels_equivalent']}/{m['labels_tested']} at delta {a.delta}; "
              f"widest label CI bound {ci_w:.4f}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
