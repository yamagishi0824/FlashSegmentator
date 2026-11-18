"""TensorRT backend for nnU-Net networks (no retraining, no architecture change).

network (PyTorch, trained weights) --ONNX (dynamic batch)--> TensorRT engine (.plan, cached on disk)

TensorRT >= 10 builds *strongly typed* networks only (no FP16 builder flag), so precision is
decided in the ONNX graph:
  * fp32: the network exported as is.
  * fp16: weights/activations in half, but every InstanceNorm is computed in fp32 (cast in,
          cast out) — the same policy torch.autocast uses in the original PyTorch path
          (instance_norm is on autocast's fp32 list). Conv weights are only cast to half.

`TRTModule(...)(x)` is a drop-in replacement for `network(x)` on (B,C,D,H,W) CUDA tensors.
Its activation memory is allocated through the torch caching allocator, so
torch.cuda.max_memory_allocated() covers it (engine weights are reported separately).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from pathlib import Path

import tensorrt as trt
import torch
import torch.nn as nn

EXPORT_VERSION = "two-stage-instnorm-v4"  # bump when the export recipe changes (invalidates cache)
_LOGGER = trt.Logger(trt.Logger.WARNING)
_TRT2TORCH = {trt.float32: torch.float32, trt.float16: torch.float16}


class TwoStageInstanceNorm3d(nn.Module):
    """Numerically equivalent InstanceNorm3d (biased var, same eps/affine), always computed in fp32,
    written as two-stage reductions (over H*W, then over D). Used ONLY for ONNX export:
    TensorRT 11 lowers a 1-stage per-(b,c) reduction over 128^3 voxels to low-parallelism Myelin
    kernels (~90 ms/layer on a small dev GPU); the 2-stage form exposes D-fold more parallelism.
    Mean-of-means is exact here because every (H*W) group has the same size."""

    def __init__(self, ref: nn.InstanceNorm3d):
        super().__init__()
        assert not ref.track_running_stats, "inference uses instance statistics only"
        self.eps = ref.eps
        self.weight = None if ref.weight is None else nn.Parameter(ref.weight.detach().float().clone())
        self.bias = None if ref.bias is None else nn.Parameter(ref.bias.detach().float().clone())

    def forward(self, x):
        in_dtype = x.dtype
        x = x.float()
        b, c, d, h, w = x.shape
        xf = x.reshape(b, c, d, h * w)
        mean = xf.mean(-1, keepdim=True).mean(-2, keepdim=True)                 # (b,c,1,1)
        var = (xf - mean).pow(2).mean(-1, keepdim=True).mean(-2, keepdim=True)
        y = (xf - mean) * torch.rsqrt(var + self.eps)
        y = y.reshape(b, c, d, h, w)
        if self.weight is not None:
            y = y * self.weight.view(1, -1, 1, 1, 1) + self.bias.view(1, -1, 1, 1, 1)
        return y.to(in_dtype)


def _export_copy(network: nn.Module, precision: str) -> nn.Module:
    """Copy of the trained network for export: same weights/topology; InstanceNorm swapped for
    the equivalent fp32 two-stage form; for fp16 everything else is cast to half (the policy
    torch.autocast applies in the original path: conv in fp16, instance_norm in fp32)."""
    net = copy.deepcopy(network).float()
    if precision == "fp16":
        net = net.half()
    for name, m in list(net.named_modules()):
        for cname, child in list(m.named_children()):
            if isinstance(child, nn.InstanceNorm3d):
                setattr(m, cname, TwoStageInstanceNorm3d(child))
    return net


def export_onnx(network: nn.Module, onnx_path: Path, in_shape, precision: str):
    net = _export_copy(network, precision)
    net = net.eval().cuda()
    dtype = torch.float32 if precision == "fp32" else torch.float16
    x = torch.randn(2, *in_shape, device="cuda", dtype=dtype)
    # Legacy TorchScript exporter on purpose: it keeps InstanceNorm as the native ONNX
    # `InstanceNormalization` op (-> TRT normalization layer). The dynamo exporter decomposes it
    # into Sub/Mul/ReduceMean, which TRT 11 on a small dev GPU runs ~13x slower than the whole PyTorch net
    # (~90 ms per full-res norm layer; see benchmarks/profile_trt.py).
    with torch.no_grad():
        torch.onnx.export(net, (x,), str(onnx_path), dynamo=False, input_names=["x"],
                          output_names=["y"], dynamic_axes={"x": {0: "b"}, "y": {0: "b"}},
                          opset_version=17, do_constant_folding=True)
    del net
    torch.cuda.empty_cache()


def build_engine(onnx_path: Path, engine_path: Path, in_shape, max_batch: int, opt_batch: int,
                 workspace_gb: float = 8.0):
    builder = trt.Builder(_LOGGER)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(net, _LOGGER)
    if not parser.parse_from_file(str(onnx_path)):
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("ONNX parse failed:\n" + "\n".join(errs))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    prof = builder.create_optimization_profile()
    prof.set_shape("x", (1, *in_shape), (opt_batch, *in_shape), (max_batch, *in_shape))
    config.add_optimization_profile(prof)
    plan = builder.build_serialized_network(net, config)
    if plan is None:
        raise RuntimeError(f"TensorRT engine build failed for {onnx_path}")
    engine_path.write_bytes(bytes(plan))


def _cache_key(state_dict, in_shape, precision, max_batch, opt_batch) -> str:
    h = hashlib.sha1()
    for k in sorted(state_dict):
        t = state_dict[k]
        h.update(k.encode()); h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().float().flatten()[:64].cpu().numpy().tobytes())
    h.update(json.dumps([EXPORT_VERSION, list(in_shape), precision, max_batch, opt_batch, trt.__version__,
                         torch.cuda.get_device_name()]).encode())
    return h.hexdigest()[:16]


class TRTModule(nn.Module):
    """Callable like the original network: y = module(x), x: (B, C, *patch) on CUDA."""

    def __init__(self, engine_path: Path):
        super().__init__()
        trt.init_libnvinfer_plugins(_LOGGER, "")
        self.runtime = trt.Runtime(_LOGGER)
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        self.engine_bytes = Path(engine_path).stat().st_size
        self.ctx = self.engine.create_execution_context(
            trt.ExecutionContextAllocationStrategy.USER_MANAGED)
        self.in_dtype = _TRT2TORCH[self.engine.get_tensor_dtype("x")]
        self.out_dtype = _TRT2TORCH[self.engine.get_tensor_dtype("y")]
        self.max_batch = self.engine.get_tensor_profile_shape("x", 0)[2][0]
        self._scratch = None
        self.stream = torch.cuda.Stream()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] > self.max_batch:   # split oversize batches instead of failing
            return torch.cat([self(c) for c in x.split(self.max_batch)])
        x = x.to(self.in_dtype).contiguous()
        self.ctx.set_input_shape("x", tuple(x.shape))
        need = self.ctx.update_device_memory_size_for_shapes()
        if self._scratch is None or self._scratch.numel() < need:
            self._scratch = torch.empty(max(need, 1), dtype=torch.uint8, device=x.device)
        self.ctx.set_device_memory(self._scratch.data_ptr(), need)
        y = torch.empty(tuple(self.ctx.get_tensor_shape("y")), dtype=self.out_dtype, device=x.device)
        self.ctx.set_tensor_address("x", x.data_ptr())
        self.ctx.set_tensor_address("y", y.data_ptr())
        cur = torch.cuda.current_stream()
        self.stream.wait_stream(cur)            # dedicated non-default stream for enqueueV3
        ok = self.ctx.execute_async_v3(self.stream.cuda_stream)
        cur.wait_stream(self.stream)
        if not ok:
            raise RuntimeError("TensorRT execution failed")
        return y

    def release_scratch(self):
        self._scratch = None


def get_trt_module(network: nn.Module, state_dict, in_shape, precision: str = "fp16",
                   max_batch: int = 8, opt_batch: int = 8,
                   cache_dir: str | os.PathLike = None) -> tuple[TRTModule, dict]:
    """Return (TRTModule, timing). Builds + caches the engine on first use, reuses afterwards."""
    cache_dir = Path(cache_dir or Path(__file__).resolve().parent.parent / "trt_engines")
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(state_dict, in_shape, precision, max_batch, opt_batch)
    engine_path = cache_dir / f"{key}_{precision}_b{max_batch}.plan"
    timing = {"engine_cached": engine_path.exists(), "onnx_export_s": 0.0, "engine_build_s": 0.0}
    if not engine_path.exists():
        onnx_dir = cache_dir / f"onnx_{key}_{precision}"
        onnx_dir.mkdir(exist_ok=True)
        onnx_path = onnx_dir / "model.onnx"
        t0 = time.perf_counter()
        export_onnx(network, onnx_path, in_shape, precision)
        timing["onnx_export_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        build_engine(onnx_path, engine_path, in_shape, max_batch, opt_batch)
        timing["engine_build_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    mod = TRTModule(engine_path)
    timing["engine_load_s"] = time.perf_counter() - t0
    timing["engine_path"] = str(engine_path)
    return mod, timing
