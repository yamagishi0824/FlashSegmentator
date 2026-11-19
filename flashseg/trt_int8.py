"""EXPERIMENTAL: TensorRT backends for the sliding-window networks (no retraining).

  fp16 : TensorRT engine, fp16 conv / fp32 InstanceNorm (= torch.autocast policy).  Baseline for int8.
  int8 : post-training quantization (symmetric INT8, per-channel weights, per-tensor activations,
         max calibration) on calibration patches recorded while running the
         normal pipeline on CALIBRATION CTs (TotalSegmentator train split, never the test set);
         Q/DQ nodes exported to ONNX, strongly typed TensorRT engine; non-quantized ops in fp16.

Engines are cached on disk (key: weights, precision, calibration tag, batch, TRT/GPU).
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
import trt_backend as TB  # noqa: E402


class _Recorder(nn.Module):
    """Pass-through that keeps a reservoir sample of input patches (CPU, fp16)."""

    def __init__(self, net, n_keep, seed):
        super().__init__()
        self.net, self.n_keep, self.seen, self.buf = net, n_keep, 0, []
        self.rng = random.Random(seed)

    def forward(self, x):
        for p in x:
            self.seen += 1
            if len(self.buf) < self.n_keep:
                self.buf.append(p.detach().half().cpu())
            else:
                j = self.rng.randrange(self.seen)
                if j < self.n_keep:
                    self.buf[j] = p.detach().half().cpu()
        return self.net(x)


def record_calibration(preds, calib_files, cfg, task, batch, n_keep=128, seed=0):
    import batched as B
    for tid, p in preds.items():
        p.network = _Recorder(p.network.eval(), n_keep, seed + tid)
    for i in range(0, len(calib_files), 4):
        B.run_group(calib_files[i:i + 4], preds, cfg, task, batch, out_dir=None, fast_post=True)
    out = {}
    for tid, p in preds.items():
        out[tid] = torch.stack(p.network.buf)
        print(f"  calib model {tid}: kept {len(p.network.buf)} of {p.network.seen} patches", flush=True)
        p.network = p.network.net
    return out


class QDQConv(nn.Module):
    """conv(x) with symmetric INT8 Q/DQ on the input (per-tensor) and weight (per-output-channel).
    torch.fake_quantize_* export to ONNX QuantizeLinear/DequantizeLinear; TensorRT fuses Q/DQ+conv
    into an INT8 conv. Non-conv ops stay in the network's own dtype (fp16; InstanceNorm fp32)."""

    def __init__(self, conv, in_amax):
        super().__init__()
        self.conv = conv
        self.transposed = isinstance(conv, nn.ConvTranspose3d)
        self.register_buffer("in_scale", torch.tensor(max(float(in_amax), 1e-8) / 127.0))
        w = conv.weight.detach().float()
        self.ch_axis = 1 if self.transposed else 0
        dims = [d for d in range(w.dim()) if d != self.ch_axis]
        self.register_buffer("w_scale", (w.abs().amax(dim=dims).clamp_min(1e-8) / 127.0))

    def forward(self, x):
        dt = x.dtype
        xq = torch.fake_quantize_per_tensor_affine(x.float(), self.in_scale.float(), torch.zeros((), dtype=torch.int32, device=x.device), -128, 127)
        wq = torch.fake_quantize_per_channel_affine(self.conv.weight.float(), self.w_scale.float(),
                                                    torch.zeros(self.w_scale.shape, dtype=torch.int32, device=x.device),
                                                    self.ch_axis, -128, 127)
        b = None if self.conv.bias is None else self.conv.bias.float()
        c = self.conv
        if self.transposed:
            y = nn.functional.conv_transpose3d(xq, wq, b, c.stride, c.padding, c.output_padding, c.groups, c.dilation)
        else:
            y = nn.functional.conv3d(xq, wq, b, c.stride, c.padding, c.dilation, c.groups)
        return y.to(dt)


def _quantized_export_copy(network, patches, calib_bs=4):
    """Max calibration: per-tensor abs-max of every Conv3d/ConvTranspose3d input over the calibration
    patches (fp32 forward), then wrap each conv in QDQConv. Returns an fp16 export copy."""
    net = TB._export_copy(network, "fp32").eval().cuda()
    convs = {n: m for n, m in net.named_modules() if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d))}
    amax = {n: 0.0 for n in convs}
    hooks = [m.register_forward_pre_hook(lambda m, a, n=n: amax.__setitem__(n, max(amax[n], float(a[0].abs().max()))))
             for n, m in convs.items()]
    with torch.no_grad():
        for i in range(0, len(patches), calib_bs):
            net(patches[i:i + calib_bs].cuda().float())
    for h in hooks:
        h.remove()
    # nnU-Net registers the same conv under several parents (e.g. `.conv` and `.all_modules[0]`);
    # replace every reference, keyed by object identity
    repl = {id(m): QDQConv(m, amax[n]).cuda() for n, m in convs.items()}
    for parent in list(net.modules()):
        if isinstance(parent, QDQConv):
            continue
        for cname, child in list(parent.named_children()):
            if id(child) in repl:
                setattr(parent, cname, repl[id(child)])
    print(f"    quantized {len(convs)} convs", flush=True)
    return net


def build_int8_engine(network, patches, in_shape, engine_path: Path, max_batch, opt_batch):
    onnx_dir = engine_path.with_suffix("")
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / "model.onnx"
    net = _quantized_export_copy(network, patches)
    net = net.half()
    x = torch.randn(2, *in_shape, device="cuda", dtype=torch.half)
    with torch.no_grad():
        torch.onnx.export(net, (x,), str(onnx_path), dynamo=False, input_names=["x"], output_names=["y"],
                          dynamic_axes={"x": {0: "b"}, "y": {0: "b"}}, opset_version=17,
                          do_constant_folding=False)
    del net
    torch.cuda.empty_cache()
    TB.build_engine(onnx_path, engine_path, in_shape, max_batch, opt_batch)


def apply(preds, precision, batch, cfg=None, task=None, calib_files=None, calib_tag="calib",
          cache_dir=None, n_keep=128):
    """Replace every predictor.network by a TensorRT module. Returns timing info."""
    cache_dir = Path(cache_dir or Path(__file__).resolve().parent.parent / "trt_engines")
    cache_dir.mkdir(parents=True, exist_ok=True)
    info, todo = {}, {}
    for tid, p in preds.items():
        in_shape = (p.network_input_channels if hasattr(p, "network_input_channels") else 1,
                    *p.configuration_manager.patch_size)
        tag = precision if precision == "fp16" else f"int8-{calib_tag}-n{n_keep}"
        key = TB._cache_key(p.network.state_dict(), in_shape, tag, batch, batch)
        path = cache_dir / f"{key}_{tag}_b{batch}.plan"
        todo[tid] = (in_shape, path)
    t0 = time.perf_counter()
    patches = None
    if precision == "int8" and not all(path.exists() for _, path in todo.values()):
        patches = record_calibration(preds, calib_files, cfg, task, batch, n_keep)
    info["calib_record_s"] = time.perf_counter() - t0
    for tid, p in preds.items():
        in_shape, path = todo[tid]
        if not path.exists():
            t = time.perf_counter()
            if precision == "fp16":
                d = path.with_suffix("")
                d.mkdir(parents=True, exist_ok=True)
                TB.export_onnx(p.network, d / "model.onnx", in_shape, "fp16")
                TB.build_engine(d / "model.onnx", path, in_shape, batch, batch)
            else:
                build_int8_engine(p.network, patches[tid], in_shape, path, batch, batch)
            info[f"build_s_{tid}"] = time.perf_counter() - t
            print(f"  engine {tid} built in {info[f'build_s_{tid}']:.0f}s -> {path.name}", flush=True)
        p.network = TB.TRTModule(path)
    return info
