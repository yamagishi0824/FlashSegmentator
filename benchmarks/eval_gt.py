"""Dice vs ground truth for several prediction dirs (TotalSegmentator `total` labels).

python benchmarks/eval_gt.py --gt <gt_dir> --pred name=dir [name=dir ...] --out results.json [--ref original]

Per case and label: Dice over labels present in the GT of that case (a label predicted but absent
from GT is counted as a false positive only through its own Dice=0 when present in GT... i.e. we
follow the usual TotalSegmentator evaluation: score labels that exist in the GT).
Reports per condition: mean over cases of the per-case mean Dice, per-label mean Dice, and the
paired per-(case,label) difference to --ref.
"""
import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np


def case_dice(args):
    gt_path, pred_path, lut = args
    g = np.asarray(nib.load(gt_path).dataobj).astype(np.uint8)
    p = np.asarray(nib.load(pred_path).dataobj).astype(np.uint8)
    if lut is not None:
        p = lut[p]
    if g.shape != p.shape:
        return None
    labs = np.unique(g); labs = labs[labs > 0]
    # per-label counts via bincount on joint index (fast)
    n = 256
    joint = np.bincount(g.astype(np.int64).ravel() * n + p.astype(np.int64).ravel(), minlength=n * n).reshape(n, n)
    gs, ps = joint.sum(1), joint.sum(0)
    return {int(l): float(2 * joint[l, l] / (gs[l] + ps[l])) for l in labs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--pred", nargs="+", required=True); ap.add_argument("--ref", default="original")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--label_map", default=None,
                    help="json {pred_to_gt: {pred_label: gt_label}} (e.g. benchmarks/btcv_label_map.json); others -> 0")
    a = ap.parse_args()
    preds = dict(x.split("=", 1) for x in a.pred)
    lut = None
    if a.label_map:
        lut = np.zeros(256, dtype=np.uint8)
        for k, v in json.load(open(a.label_map))["pred_to_gt"].items():
            lut[int(k)] = int(v)
    res = {}
    per = {}
    for name, d in preds.items():
        cases = sorted(os.path.basename(f) for f in glob.glob(os.path.join(d, "*.nii.gz")))
        cases = [c for c in cases if os.path.exists(os.path.join(a.gt, c))]
        with ProcessPoolExecutor(a.workers) as ex:
            out = list(ex.map(case_dice, [(os.path.join(a.gt, c), os.path.join(d, c), lut) for c in cases]))
        per[name] = {c: o for c, o in zip(cases, out) if o}
        case_means = [np.mean(list(o.values())) for o in per[name].values() if o]
        lab = {}
        for o in per[name].values():
            for l, v in o.items():
                lab.setdefault(l, []).append(v)
        res[name] = {"n_cases": len(per[name]), "dice_case_mean": float(np.mean(case_means)),
                     "dice_case_median": float(np.median(case_means)),
                     "dice_label_mean": float(np.mean([np.mean(v) for v in lab.values()])),
                     "per_label": {int(l): float(np.mean(v)) for l, v in sorted(lab.items())}}
        print(name, {k: res[name][k] for k in ("n_cases", "dice_case_mean", "dice_case_median", "dice_label_mean")}, flush=True)
    if a.ref in per:
        for name in per:
            if name == a.ref:
                continue
            diffs = [per[name][c][l] - per[a.ref][c][l] for c in per[name] if c in per[a.ref]
                     for l in per[name][c] if l in per[a.ref][c]]
            if diffs:
                d = np.array(diffs)
                res[name]["paired_vs_ref"] = {"n": int(d.size), "mean": float(d.mean()), "abs_max": float(np.abs(d).max()),
                                              "frac_nonzero": float((d != 0).mean())}
                print(" ", name, "vs", a.ref, res[name]["paired_vs_ref"], flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
