"""Output-identical faster post-processing for the `total` path (PERF_LOG §5).

5a  logits -> segmentation: the original
      convert_predicted_logits_to_segmentation_with_correct_shape
    (i) casts the fp16 logits to fp32 (torch_resample_data_or_seg_to_shape returns early on equal
    shapes but still copies to fp32), (ii) calls gc.collect() + empty_cache() per model per case,
    (iii) reverts the crop on the CPU. Here: max/threshold on the fp16 logits directly (fp16->fp32
    is exact, so max/argmax/>=0.5 give identical results), crop revert on GPU, no gc/empty_cache.
    Falls back to the original function whenever a real resample (shape change) or softmax is needed.

5b  part merge: the original loops over each part's labels in numpy
      seg_combined[part_seg == local_idx] = global_idx[name]
    Here a per-part lookup table on the GPU does the same in one gather (parts are applied in the
    same order, so later parts overwrite earlier ones exactly as before).
"""
from __future__ import annotations

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice

import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent / "fastsegmentator"))
import totalseg_infer as ti  # noqa: E402
from nnunet_infer_nii import convert_predicted_logits_to_segmentation_with_correct_shape as _orig_convert


def logits_to_seg_gpu(predicted_logits, plans_manager, configuration_manager, label_manager,
                      props, use_softmax):
    """Returns the segmentation as a GPU tensor (uint8/uint16), identical to the original's output."""
    target_shape = list(props['shape_after_cropping_and_before_resampling'])
    if use_softmax or list(predicted_logits.shape[1:]) != [int(s) for s in target_shape]:
        return _orig_convert(predicted_logits, plans_manager, configuration_manager, label_manager,
                             props, use_softmax).to(predicted_logits.device)
    max_logit, max_class = torch.max(predicted_logits, dim=0)
    dtype = torch.uint8 if len(label_manager.foreground_labels) < 255 else torch.uint16
    seg = torch.where(max_logit >= 0.5, max_class, torch.zeros((), dtype=max_class.dtype,
                                                                  device=max_class.device)).to(dtype)
    out = torch.zeros(tuple(props['shape_before_cropping']), dtype=dtype, device=seg.device)
    out[bounding_box_to_slice(props['bbox_used_for_cropping'])] = seg
    return out.permute(plans_manager.transpose_backward)


_LUT_CACHE: dict = {}


def merge_lut(task_id, cfg, device):
    """(lut, valid): lut[local] = global label; valid[local] = local label is mapped by this part."""
    key = (task_id, cfg.class_map_key, cfg.class_map_parts_key, str(device))
    if key not in _LUT_CACHE:
        gidx = {n: i for i, n in ti.class_map[cfg.class_map_key].items()}
        lut = torch.zeros(256, dtype=torch.uint8); valid = torch.zeros(256, dtype=torch.bool)
        if cfg.class_map_parts_key is None:
            lut = torch.arange(256, dtype=torch.int32).to(torch.uint8); valid[:] = True
        else:
            parts = ti._CLASS_MAP_PARTS[cfg.class_map_parts_key]
            pname = ti._PARTNAME_MAPS[cfg.partname_map_key][task_id]
            for local, name in parts[pname].items():
                lut[local] = gidx[name]; valid[local] = True
        _LUT_CACHE[key] = (lut.to(device), valid.to(device))
    return _LUT_CACHE[key]


def merge_part_gpu(part_seg: torch.Tensor, task_id, cfg, seg_combined: torch.Tensor):
    """In place on GPU; same semantics as ti._merge_part (multi-part) / np.copyto (single-part)."""
    lut, valid = merge_lut(task_id, cfg, part_seg.device)
    idx = part_seg.long()
    if cfg.class_map_parts_key is None:
        seg_combined.copy_(part_seg.to(torch.uint8))
    else:
        m = valid[idx]
        seg_combined[m] = lut[idx][m]
    return seg_combined
