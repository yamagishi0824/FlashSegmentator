# FlashSegmentator

Throughput-oriented inference for the **TotalSegmentator `total` task** (CT, 117 structures, 5 part-models)
and the **`total_mr` task** (MRI, 50 structures, 2 part-models; `--mri`),
built on top of **[FastSegmentator](https://github.com/bowang-lab/FastSegmentator)** (bowang-lab), which is itself a
fast re-implementation of the inference path of **[TotalSegmentator](https://github.com/wasserth/TotalSegmentator)**
(Wasserthal et al.) / **[nnU-Net](https://github.com/MIC-DKFZ/nnUNet)** (Isensee et al.).

- Same pretrained TotalSegmentator weights, same network, same sliding-window settings
  (patch 128³, step 0.8, Gaussian weighting, no mirroring, fp16 autocast) — **no retraining**.
- Same accuracy: Dice vs ground truth matches official TotalSegmentator to within ±0.00002 (fp16 summation-order noise;
  all Dice values in this README are given to 5 decimals).
- Faster when processing many CTs: on GH200, **~3.4× FastSegmentator** (3.2× FastSegmentator with its `nnUNet_compile=1`)
  and **~58× the official TotalSegmentator CLI**.

> Status: research code, work in progress. Only `total` (CT) and `total_mr` (MRI, `--mri`) are covered. The original FastSegmentator README
> (all other tasks/modes, parity tables) is kept in [`docs/fastsegmentator/README_FastSegmentator.md`](docs/fastsegmentator/README_FastSegmentator.md).

## Results (TotalSegmentator v2.0.1 test split, 73 CTs, NVIDIA GH200)

| | per CT (mean, batch run) | throughput | speed vs TotalSegmentator | Dice vs GT (case mean) |
|---|---|---|---|---|
| TotalSegmentator (official CLI, one call per CT) | 55.0 s | 65.5 CT/h | 1.0× | 0.93765 |
| FastSegmentator (`totalseg_infer.py`, upstream) | 3.26 s | 1104 CT/h | 16.9× | 0.93765 |
| FastSegmentator + its `nnUNet_compile=1` | 3.05 s | 1182 CT/h | 18.0× | 0.93765 |
| **FlashSegmentator** (batch 8) | **0.95 s** | **3782 CT/h** | **57.7×** | **0.93766** |
| FlashSegmentator (batch 8) `--body_crop` | 0.77 s | 4695 CT/h | 71.7× | 0.93880¹ |

¹ Non-inferior to TotalSegmentator overall (+0.0012, 90% CI 0.0002–0.0021; margin −0.01) and for 106/117 structures; exceptions include some ribs, C5–C6 and kidney cysts. Use the default (no crop) for whole-body quantification of thin structures.

External check, BTCV (30 CTs, 13 abdominal labels mapped from TS labels): FastSegmentator 975 CT/h → FlashSegmentator
3333 CT/h (batch 8; 3.42×), single-CT latency 3.57 s → 1.63 s, BTCV Dice 0.84463 for both. With `--body_crop`: 0.8447 (difference vs no crop +0.0001, 90% CI −0.0009 to 0.0011).

Smaller GPU — NVIDIA T4 16 GB (Kaggle), BTCV 12 CTs:

| | per CT (mean, batch run) | throughput | speed vs TotalSegmentator | BTCV Dice |
|---|---|---|---|---|
| TotalSegmentator (official CLI, one call per CT) | 135.9 s | 26.5 CT/h | 1.0× | 0.81903 |
| FastSegmentator (upstream) | 25.6 s | 141 CT/h | 5.3× | 0.81902 |
| FastSegmentator + its `nnUNet_compile=1` | 22.0 s | 164 CT/h | 6.2× | 0.81902 |
| **FlashSegmentator** (batch 2) | **17.5 s** | **206 CT/h** | **7.8×** | **0.81902** |

On a slow GPU the network itself dominates, so the gain over FastSegmentator is smaller (1.46×; 1.26× over FastSegmentator with `nnUNet_compile=1`) and larger batches
do not help (use `--batch 2`).
Timings exclude model load (~3.5 s) and the one-off `torch.compile` warm-up (~15–30 s), which are reported separately
in the log. Note that the official-CLI number also includes per-call start-up and model loading, so it measures
"TotalSegmentator as normally used", not raw network speed.

## Concurrent processes on one GPU (512 TotalSegmentator v2.0.1 training-split CTs)

Best configuration of each implementation (number of concurrent processes on one GPU in parentheses).
Throughput = 512 CTs / time of the longest-running process; a configuration in which any process ran out of GPU memory was not counted.

| | GH200 | A100 80GB PCIe |
|---|---|---|
| TotalSegmentator (official CLI, `xargs -P N`) | not evaluated¹ | 482 CT/h (32) |
| FastSegmentator + `nnUNet_compile=1` | 3079 CT/h (4) | 1707 CT/h (4) |
| **FlashSegmentator** (GH200 batch 8, A100 batch 4) | **4707 CT/h (1)** | **2263 CT/h (1)** |
| FlashSegmentator `--body_crop` | 6023 CT/h (1) | 2931 CT/h (1) |

FlashSegmentator is fastest with a single process: it already overlaps CPU pre/post-processing with GPU inference, and
extra processes only compete for GPU memory. For more throughput, run one process per GPU rather than several per GPU.
¹ On the GH200 node, 16 or more concurrent official CLI calls failed because of multiprocessing limits.

## MRI (`--mri`, TotalSegmentator MRI v2.0.0 test split, 55 MRIs, NVIDIA A100 80GB PCIe)

| | per MRI (mean, batch run) | throughput | speed vs TotalSegmentator | Dice vs GT (case mean, 49 MRIs) |
|---|---|---|---|---|
| TotalSegmentator (official CLI `--task total_mr`, one call per MRI) | 25.8 s | 140 /h | 1.0× | 0.78157 |
| **FlashSegmentator `--mri`** (batch 4) | **0.27 s** | **13,139 /h** | **94×** | **0.78158** |

Paired equivalence vs the official CLI (TOST, ±0.01 Dice): all 50 structures equivalent (widest 90% CI bound 0.0010).
6 of the 55 test MRIs have empty reference masks in the dataset; they are timed but not included in Dice.
For MRI, Flash resamples exactly as the official TotalSegmentator does (scipy `ndimage.zoom` grid, linear input
interpolation; computed on the GPU with `cupyx.scipy.ndimage.zoom`) and uses tile step 0.8 like the official CLI.
FastSegmentator's MRI path differed in both (cucim resize grid, cubic interpolation, step 0.5), which gave
structure-level Dice differences up to ~0.2 vs the official CLI. The CT path is unchanged.

Every experiment, including the ones that did **not** help, is recorded with numbers in
[`docs/PERF_LOG.md`](docs/PERF_LOG.md).

## What is different from FastSegmentator

FastSegmentator processes one CT at a time, one 128³ patch at a time. FlashSegmentator keeps its preprocessing,
networks and postprocessing semantics and changes how the work is scheduled:

| change | where | effect (GH200, TS-73) |
|---|---|---|
| **Load the 5 models once** for a whole folder, process CTs back to back | `flashseg/batched.py` | removes per-CT start-up |
| **Batched sliding window**: patches from several CTs ("group") are pooled and run `B` at a time; OOM ⇒ batch halves automatically | `flashseg/batched.py` (`predict_group`, `_forward_oom_safe`) | — |
| **3-stage pipeline**: CPU threads load/resample the next group and write the previous one while the GPU runs the current group | `flashseg/batched.py` (`run_pipeline`) | 1104 → 1670 CT/h |
| **logits → labels on the GPU** (fp16 argmax/threshold, crop revert, LUT-based part merge), no fp32 host copies, no `gc.collect()` per CT | `flashseg/fastpost.py` | → 2283 CT/h |
| **Faster writer**: nibabel header/affine reuse, uint8, no re-read of the input (also fixes inputs that SimpleITK cannot read: non-orthonormal direction cosines) | `flashseg/batched.py` (`write_case_fast`) | (part of the above) |
| **Static-shape `torch.compile`**: one graph per model at a fixed batch size, tail batches zero-padded (FastSegmentator's `nnUNet_compile=1` recompiles dynamically when the batch varies); dynamo cache limit raised so 5 models × several batch sizes never fall back to eager | `flashseg/static_compile.py` | → 3541 (B2) / 3782 (B8) CT/h |
| `cudnn.benchmark` on | `flashseg/cli.py` | same speed, compile warm-up 40 s → 14 s |

Unchanged: model architecture and weights, ensemble of the 5 part-models, patch size, step size/overlap, Gaussian
weighting, resampling/interpolation, normalization. Outputs differ from FastSegmentator only by fp16 summation-order
noise (~100–250 voxels per CT, the same size as run-to-run nondeterminism of the original).

Tried and **not** adopted (details in the log): TensorRT FP16/FP32 (slower than `torch.compile` on GH200;
InstanceNorm becomes slow reductions), ONNX Runtime, fusing the 5 networks into one grouped-conv network
(faster in eager, slower once compiled), `channels_last_3d` / Inductor layout optimisation (cuDNN's NCHW↔NHWC
transposes, ~1/3 of the forward, remain either way), larger batches on small GPUs.

## Repository layout

```
flashseg/          FlashSegmentator: cli.py, batched.py (batching + pipeline), fastpost.py, static_compile.py
fastsegmentator/   upstream FastSegmentator inference code (totalseg_infer.py, nnunet_infer_nii.py, cli.py, ...)
src/               vendored nnunetv2 (upstream)
benchmarks/        benchmark, profiling and evaluation scripts (bench2.py, eval_gt.py, probes, TensorRT/fused trials)
docs/              PERF_LOG.md (every experiment) and the upstream FastSegmentator docs
tests/, report/    upstream
```

## Usage

Set up the environment as described in [`docs/fastsegmentator/README_FastSegmentator.md`](docs/fastsegmentator/README_FastSegmentator.md) (TotalSegmentator
checkout as a sibling, weights via `fastsegmentator/download_totalseg_weights.py`), then:

```bash
python flashseg/cli.py -i /path/to/ct_folder -o /path/to/out --batch 8     # large GPU (H100/GH200)
python flashseg/cli.py -i /path/to/ct_folder -o /path/to/out --batch 2     # smaller GPU
python flashseg/cli.py -i ct.nii.gz -o out --no_compile                    # skip torch.compile warm-up
python flashseg/cli.py -i /path/to/mri_folder -o /path/to/out --mri        # MRI: TotalSegmentator `total_mr` (50 structures)
python flashseg/cli.py -i /path/to/ct_folder -o /path/to/out --body_crop   # optional, CT only (see below)
```

- `--mri` runs the TotalSegmentator `total_mr` task (part-models 850 and 851, 1.5 mm) with the same batched pipeline.
  Download its weights first (e.g. `totalseg_download_weights -t total_mr`, or the same layout under `--weights`).
- `--body_crop` crops each resampled CT to the bounding box of the body (HU > −500) before tiling (~25% fewer patches,
  ~1.3x faster). **It changes the output slightly** (outside-body voxels are written as background), so it is off by
  default; it uses a CT intensity threshold and cannot be combined with `--mri`.

Run log (only when `--log` is given):

```bash
python flashseg/cli.py -i cts/ -o out/ --log                 # log -> out/flashseg_log/{summary.json,per_case.csv}
python flashseg/cli.py -i cts/ -o out/ --log runs/gh200_b8   # log to another directory
```

Example `summary.json`:

```json
{
 "n_cases": 73,
 "settings": {"batch": 8, "group": 4, "compile": true, "cudnn_benchmark": true, "task": "total"},
 "model_load_s": 3.6,
 "first_case_incl_compile_s": 30.9,
 "throughput_ct_per_hour": 3770,
 "mean_s_per_ct_excl_first": 0.95,
 "total_wall_s": 100.4,
 "peak_gpu_mem_gb": 15.8,
 "gpu": "NVIDIA GH200 120GB",
 "torch": "2.11.0+cu128",
 "flashsegmentator_commit": "09fe6e3"
}
```

`per_case.csv` has one row per CT: `case,done_at_s,interval_s` (seconds since the first CT started; in a pipeline,
CTs of one group finish together, so intervals are bursty).

One multilabel `.nii.gz` per input is written (TotalSegmentator `total` label map, or `total_mr` with `--mri`).
With `--log`, a run log is written to `<out>/flashseg_log/` (or `--log DIR`):
`summary.json` (throughput excluding the first CT, model-load time, first-CT time incl. `torch.compile` warm-up,
peak GPU memory, GPU / torch / CUDA versions, settings, commit) and `per_case.csv` (completion time of each CT).
Throughput is meaningful for folders of more than a few CTs (the pipeline needs a few groups to fill).
Benchmark / evaluation scripts: `benchmarks/bench2.py` (throughput, latency, memory), `benchmarks/eval_gt.py` (Dice vs GT).

## Acknowledgements and license

- **FastSegmentator** — https://github.com/bowang-lab/FastSegmentator (Apache-2.0). This repository is a fork; its
  history is preserved. Upstream code lives unchanged in `fastsegmentator/` (only the relative path to the
  TotalSegmentator checkout was adjusted after moving it there) and its docs in `docs/fastsegmentator/`.
- **TotalSegmentator** — https://github.com/wasserth/TotalSegmentator. Please cite: Wasserthal J. et al.,
  *TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images*, Radiology: AI, 2023.
  The pretrained weights are TotalSegmentator's and are subject to its license terms.
- **nnU-Net** — Isensee F. et al., *nnU-Net: a self-configuring method for deep learning-based biomedical image
  segmentation*, Nature Methods, 2021.

Licensed under Apache-2.0 (see `LICENSE`), same as FastSegmentator.
