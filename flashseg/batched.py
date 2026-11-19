"""Batched sliding-window inference for FastSegmentator `total` (and other uncropped modes).

Patches from several CTs are pooled into one queue and run through the network in batches of
`batch_size`; logits are scattered back into each CT's own accumulator (same per-patch ops,
same patch order within a case, same gaussian weighting / fp16 accumulators as the original).
Pre-processing and post-processing are the ORIGINAL functions / code paths of totalseg_infer.py;
only the split into prepare → forward → finish is new (needed to interleave cases).

OOM guard: on torch.OutOfMemoryError the batch is halved and retried (down to 1).
"""
from __future__ import annotations

import sys
import time
_TIMERS = __import__('os').environ.get('FS_TIMERS') == '1'  # PERF_LOG §12: synced per-stage timers (diagnosis only)
from pathlib import Path

import numpy as np
import torch
import nibabel as nib
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parent.parent / "fastsegmentator"   # upstream FastSegmentator code
sys.path.insert(0, str(ROOT))
import totalseg_infer as ti  # noqa: E402
from nnunet_infer_nii import (  # noqa: E402
    convert_predicted_logits_to_segmentation_with_correct_shape, _undo_canonical)

# Optional body crop (off by default; output no longer bit-near the original, see PERF_LOG §20).
# After resampling, the image is cropped to the bounding box of the body (HU > BODY_HU, couch removed
# by opening, components >= 1% of the largest kept so separated arms stay) + BODY_MARGIN voxels;
# everything outside the box is written as background.
BODY_CROP = False
BODY_HU, BODY_MARGIN = -500.0, 10
# MRI (`total_mr`) only: resample exactly like the reference TotalSegmentator (commit 8806051): scipy grid
# (change_spacing_ref), linear input interpolation. The CT path (`total`) is unchanged (FastSegmentator behaviour,
# cucim GPU resampling, cubic input interpolation): the CT code path is identical to earlier versions.
MR_RESAMPLE_ORDER = 1


def _body_bbox(img_zyx):
    """Bounding box (lo, hi) in voxels of the body on the resampled image (Z, Y, X); None = no crop."""
    from scipy.ndimage import binary_opening, label
    m = img_zyx[::2, ::2, ::2] > BODY_HU                        # half resolution is enough for a box
    m = binary_opening(m, iterations=2)                          # drops couch / thin straps
    lab, n = label(m)
    if n == 0:
        return None
    sizes = np.bincount(lab.ravel())[1:]
    keep = np.isin(lab, np.nonzero(sizes >= 0.01 * sizes.max())[0] + 1)
    idx = np.argwhere(keep)
    lo = np.maximum(idx.min(0) * 2 - BODY_MARGIN, 0)
    hi = np.minimum(idx.max(0) * 2 + 2 + BODY_MARGIN, img_zyx.shape)
    return lo, hi
def change_spacing_ref(img_in, new_spacing, target_shape=None, order=0, dtype=None, force_affine=None):
    """= totalsegmentator.resampling.change_spacing on its CPU path (scipy.ndimage.zoom, mode="nearest"),
    which is what the reference TotalSegmentator uses unless cucim is installed, but computed on the GPU with
    cupyx.scipy.ndimage.zoom (same sampling grid, float64). TotalSegmentator's own GPU path (cucim
    skimage.resize) uses a pixel-centre grid and gives different values whenever the spacing changes."""
    import cupy as cp
    from cupyx.scipy import ndimage as cnd
    data = img_in.get_fdata()
    old_shape = np.array(data.shape)
    img_spacing = np.array(img_in.header.get_zooms())[:3]
    new_spacing = np.array([new_spacing] * 3 if np.isscalar(new_spacing) else new_spacing, dtype=float)
    if target_shape is not None:
        zoom = np.array(target_shape) / old_shape
        new_spacing = img_spacing / zoom
    else:
        zoom = img_spacing / new_spacing
    if np.array_equal(img_spacing, new_spacing):
        return img_in
    new_affine = np.copy(img_in.affine)
    for k in range(3):
        new_affine[:3, k] = new_affine[:3, k] / zoom[k]
    new_data = cnd.zoom(cp.asarray(data, dtype=cp.float64), zoom, mode="nearest", order=order).get()
    if dtype is not None:
        new_data = new_data.astype(dtype)
    if force_affine is not None:
        new_affine = force_affine
    return nib.Nifti1Image(new_data, new_affine)


from acvl_utils.cropping_and_padding.padding import pad_nd_image  # noqa: E402
from nnunetv2.inference.sliding_window_prediction import compute_gaussian  # noqa: E402


def prepare_case(file_path, cfg):
    """= first half of ti._predict_to_original_space (no crop modes)."""
    assert not cfg.crop, "batched path supports uncropped modes only"
    img_orig = nib.load(file_path)
    img_can = nib.as_closest_canonical(img_orig)
    if cfg.resample is not None:
        if cfg.modality == "mr":
            img_rsp = change_spacing_ref(img_can, list(cfg.resample), order=MR_RESAMPLE_ORDER, dtype=np.int32)
        else:
            img_rsp = ti.tseg_change_spacing(img_can, list(cfg.resample), order=3, dtype=np.int32, use_gpu=True)
        arr_xyz = img_rsp.get_fdata().astype(np.float32)
        spacing_zyx = [float(cfg.resample[2]), float(cfg.resample[1]), float(cfg.resample[0])]
    else:
        img_rsp = img_can
        arr_xyz = img_can.get_fdata().astype(np.float32)
        z = img_can.header.get_zooms()[:3]
        spacing_zyx = [float(z[2]), float(z[1]), float(z[0])]
    image = arr_xyz.transpose(2, 1, 0)[np.newaxis]
    bbox = _body_bbox(image[0]) if BODY_CROP else None
    if bbox is not None:
        full_shape = image.shape[1:]
        image = np.ascontiguousarray(image[(slice(None), *[slice(a, b) for a, b in zip(*bbox)])])
        bbox = (bbox[0], bbox[1], full_shape)
    return dict(file=file_path, img_orig=img_orig, img_can=img_can, img_can_shape=img_can.shape,
                img_rsp=img_rsp, image=image, bbox=bbox,
                props={'spacing': spacing_zyx}, seg=None)


def finish_case(c, cfg, task_name):
    """= second half of ti._predict_to_original_space (postprocess at resample res, undo)."""
    if c.get('bbox') is not None:                              # body crop: paste back into full grid
        lo, hi, full = c['bbox']
        seg = np.zeros(full, dtype=np.uint8)
        seg[tuple(slice(a, b) for a, b in zip(lo, hi))] = c['seg']
        c['seg'] = seg
    seg_nib = nib.Nifti1Image(c['seg'].transpose(2, 1, 0).astype(np.uint8), c['img_rsp'].affine)
    seg_nib = ti.remove_auxiliary_labels(seg_nib, task_name)
    if cfg.resample is not None:
        if cfg.modality == "mr":
            seg_nib = change_spacing_ref(seg_nib, list(cfg.resample), c['img_can_shape'], order=0,
                                         force_affine=c['img_can'].affine)
        else:
            seg_nib = ti.tseg_change_spacing(seg_nib, list(cfg.resample), c['img_can_shape'], order=0,
                                             force_affine=c['img_can'].affine, use_gpu=True)
    return ti.tseg_undo_canonical(seg_nib, c['img_orig'])


def write_case(seg_nib, file_path, out_dir):
    """= output part of ti.infer_file."""
    img_data = seg_nib.get_fdata().astype(np.uint8)
    name = Path(file_path).name.replace("_0000.nii.gz", "").replace(".nii.gz", "")
    s = sitk.GetImageFromArray(img_data.transpose(2, 1, 0).astype(np.uint8))
    s.CopyInformation(sitk.ReadImage(file_path))
    sitk.WriteImage(s, str(Path(out_dir) / f"{name}.nii.gz"))
    return img_data


def _forward_oom_safe(net, x, state):
    """Run net on x in chunks of state['bs']; halve on OOM (persists for later calls)."""
    outs, i = [], 0
    while i < x.shape[0]:
        bs = state['bs']
        try:
            outs.append(net(x[i:i + bs]))
            i += bs
        except torch.OutOfMemoryError:
            if bs == 1:
                raise
            torch.cuda.empty_cache()
            state['bs'] = bs // 2
            state['oom_events'] += 1
            print(f"  [OOM] batch {bs} -> {state['bs']}", flush=True)
    return torch.cat(outs) if len(outs) > 1 else outs[0]


@torch.no_grad()
def predict_group(predictor, cases, use_softmax, batch_size, state=None, fast_post=False):
    """One model over a group of cases; returns per-case segmentation (numpy, model label space).
    Mirrors SimplePredictor.inference + _internal_predict_sliding_window_return_logits."""
    state = state if state is not None else {'bs': batch_size, 'oom_events': 0}
    dev = predictor.device
    net = predictor.network.to(dev).eval()
    ps = tuple(predictor.configuration_manager.patch_size)
    nh = predictor.label_manager.num_segmentation_heads
    with torch.autocast(dev.type, enabled=True):
        gaussian = compute_gaussian(ps, sigma_scale=1. / 8, value_scaling_factor=10, device=dev)
        work = []
        for c in cases:
            img = predictor.preprocess(c['image'], c['props'])
            data, revert = pad_nd_image(img, ps, 'constant', {'value': 0}, True, None)
            slicers = predictor._internal_get_sliding_window_slicers(data.shape[1:])
            work.append(dict(data=data.to(dev), revert=revert,
                             logits=torch.zeros((nh, *data.shape[1:]), dtype=torch.half, device=dev),
                             npred=torch.zeros(data.shape[1:], dtype=torch.half, device=dev)))
            for sl in slicers:
                work[-1].setdefault('queue', []).append(sl)
        # global queue: case-major order keeps each case's patch order identical to the original
        queue = [(ci, sl) for ci, w in enumerate(work) for sl in w['queue']]
        for i in range(0, len(queue), batch_size):
            chunk = queue[i:i + batch_size]
            if _TIMERS: torch.cuda.synchronize(); _t0 = time.perf_counter()
            x = torch.stack([work[ci]['data'][sl] for ci, sl in chunk])
            if _TIMERS: torch.cuda.synchronize(); _t1 = time.perf_counter()
            y = _forward_oom_safe(net, x, state)
            if _TIMERS: torch.cuda.synchronize(); _t2 = time.perf_counter()
            for (ci, sl), pred in zip(chunk, y):
                pred = pred * gaussian
                work[ci]['logits'][sl] += pred
                work[ci]['npred'][sl[1:]] += gaussian
            if _TIMERS:
                torch.cuda.synchronize(); _t3 = time.perf_counter()
                for k, v in (('t_stack_s', _t1 - _t0), ('t_fwd_s', _t2 - _t1), ('t_accum_s', _t3 - _t2)):
                    state[k] = state.get(k, 0.0) + v
                state['n_patches'] = state.get('n_patches', 0) + len(chunk)
                state['n_calls'] = state.get('n_calls', 0) + 1
        segs = []
        for c, w in zip(cases, work):
            lg = w['logits']; lg /= w['npred']
            del w['npred']
            # per-channel check: a full-volume isinf() mask needed GB of extra memory and OOMed on large
            # CTs (T4 / several processes per GPU), outside the forward pass's OOM guard
            if any(bool(torch.isinf(lg[k]).any()) for k in range(lg.shape[0])):
                raise RuntimeError("inf in predicted logits")
            lg = lg[(slice(None), *w['revert'][1:])]
            del w['data']
            if fast_post:   # PERF_LOG 5a: fp16 max/threshold + crop revert on GPU, stays on GPU
                import fastpost
                try:
                    seg = fastpost.logits_to_seg_gpu(lg, predictor.plans_manager, predictor.configuration_manager,
                                                     predictor.label_manager, c['props'], use_softmax)
                except torch.OutOfMemoryError:   # free cached blocks once and retry (large CTs, shared GPU)
                    torch.cuda.empty_cache(); state['oom_events'] += 1
                    seg = fastpost.logits_to_seg_gpu(lg, predictor.plans_manager, predictor.configuration_manager,
                                                     predictor.label_manager, c['props'], use_softmax)
            else:
                seg = convert_predicted_logits_to_segmentation_with_correct_shape(
                    lg, predictor.plans_manager, predictor.configuration_manager, predictor.label_manager,
                    c['props'], use_softmax, return_probabilities=False)
            if c['props'].get('_can_fwd_flips') is not None:
                seg = _undo_canonical(seg, c['props']['_can_fwd_flips'], c['props']['_can_inv_perm'])
            segs.append(seg if fast_post else seg.numpy())
            del lg, w['logits']
    return segs


def write_case_fast(seg_nib, img_orig, file_path, out_dir):
    """PERF_LOG 5c: no re-read of the input CT (the original calls sitk.ReadImage(file_path) only for
    CopyInformation, which also crashes on non-orthonormal direction cosines), no float64 get_fdata.
    Saves with the input's own nibabel affine/header (uint8)."""
    img_data = np.asarray(seg_nib.dataobj).astype(np.uint8)
    name = Path(file_path).name.replace("_0000.nii.gz", "").replace(".nii.gz", "")
    hdr = img_orig.header.copy(); hdr.set_data_dtype(np.uint8)
    hdr["scl_slope"] = 1; hdr["scl_inter"] = 0
    out = nib.Nifti1Image(img_data, img_orig.affine, hdr)
    nib.save(out, str(Path(out_dir) / f"{name}.nii.gz"))
    return img_data


def _forward_and_merge(cases, predictors, cfg, batch_size, state, fast_post):
    gidx = {n: i for i, n in ti.class_map[cfg.class_map_key].items()}
    for tid, pred in predictors.items():
        for c, part in zip(cases, predict_group(pred, cases, cfg.use_softmax, batch_size, state, fast_post)):
            if fast_post:   # PERF_LOG 5b: LUT merge on GPU
                import fastpost
                if c['seg'] is None:
                    c['seg'] = torch.zeros(part.shape, dtype=torch.uint8, device=part.device)
                fastpost.merge_part_gpu(part, tid, cfg, c['seg'])
            else:
                if c['seg'] is None:
                    c['seg'] = np.zeros(part.shape, dtype=np.uint8)
                ti._merge_part(part, tid, cfg, c['seg'], gidx)
    if fast_post:
        for c in cases:
            c['seg'] = c['seg'].cpu().numpy()


def run_group(files, predictors, cfg, task_name, batch_size, out_dir=None, state=None, fast_post=False):
    """Full pipeline for a group of CTs sharing GPU batches. Returns list of label arrays (XYZ)."""
    cases = [prepare_case(f, cfg) for f in files]
    _forward_and_merge(cases, predictors, cfg, batch_size, state, fast_post)
    out = []
    for c in cases:
        seg_nib = finish_case(c, cfg, task_name)
        out.append(write_case(seg_nib, c['file'], out_dir) if out_dir
                   else seg_nib.get_fdata().astype(np.uint8))
    return out


def run_pipeline(files, predictors, cfg, task_name, batch_size, group, out_dir,
                 n_prep=2, n_write=2, prefetch_groups=1, state=None, on_done=None, fast_post=False):
    """Same computation as run_group over consecutive groups, but overlapped:
    prep thread(s) load/resample group g+1 while the main thread runs the GPU forward of group g,
    and writer thread(s) undo-resample + gzip-write group g-1. Bounded by `prefetch_groups`."""
    from concurrent.futures import ThreadPoolExecutor
    groups = [files[i:i + group] for i in range(0, len(files), group)]
    gidx = {n: i for i, n in ti.class_map[cfg.class_map_key].items()}

    def write(c):
        seg_nib = finish_case(c, cfg, task_name)
        if fast_post:
            write_case_fast(seg_nib, c['img_orig'], c['file'], out_dir)
        else:
            write_case(seg_nib, c['file'], out_dir)
        if on_done:
            on_done(c['file'])
        return c['file']

    with ThreadPoolExecutor(n_prep) as prep_ex, ThreadPoolExecutor(n_write) as write_ex:
        pending = [[prep_ex.submit(prepare_case, f, cfg) for f in g] for g in groups[:1 + prefetch_groups]]
        nxt, writes = 1 + prefetch_groups, []
        for gi in range(len(groups)):
            _t = time.perf_counter()
            cases = [fu.result() for fu in pending.pop(0)]          # main thread waits for prep here
            if state is not None:
                state['wait_prep_s'] = state.get('wait_prep_s', 0.0) + time.perf_counter() - _t
            if nxt < len(groups):
                pending.append([prep_ex.submit(prepare_case, f, cfg) for f in groups[nxt]]); nxt += 1
            _forward_and_merge(cases, predictors, cfg, batch_size, state, fast_post)
            for c in cases:
                c['image'] = None                      # free input early
                writes.append(write_ex.submit(write, c))
        _t = time.perf_counter()
        for w in writes:
            w.result()
        if state is not None:
            state['wait_write_tail_s'] = state.get('wait_write_tail_s', 0.0) + time.perf_counter() - _t
