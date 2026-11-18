"""Probe: single model, PyTorch-autocast vs TRT fp32/fp16, forward-only timing + numeric diff."""
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "fastsegmentator")); sys.path.insert(0, str(ROOT / "flashseg")); sys.path.insert(0, str(ROOT / "benchmarks"))
import totalseg_infer as ti  # noqa: E402
from trt_backend import get_trt_module  # noqa: E402

dev = torch.device("cuda")
cfg = ti.TASK_CONFIGS["total"]
p = ti.build_predictors(ti.replace(cfg, task_ids=(291,)), Path.home() / ".totalsegmentator/nnunet/results",
                        dev, 0.8)[291]
net = p.network.eval()
in_shape = (1, *p.configuration_manager.patch_size)


def bench(fn, x, n=10):
    for _ in range(3):
        fn(x)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n):
        fn(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n


def torch_fwd(x):
    with torch.no_grad(), torch.autocast("cuda"):
        return net(x)


mods = {}
for prec in ("fp32", "fp16"):
    m, tm = get_trt_module(net, p.list_of_parameters[-1], in_shape, prec)
    print(prec, tm, flush=True)
    mods[prec] = m
for B in (1, 4, 8):
    x = torch.randn(B, *in_shape, device=dev)
    ref = torch_fwd(x).float()
    line = f"B={B}: torch-autocast {bench(torch_fwd, x)*1e3/B:.1f} ms/patch"
    for prec, m in mods.items():
        y = m(x).float()
        line += (f" | trt-{prec} {bench(m, x)*1e3/B:.1f} ms/patch, "
                 f"max|d|={(y-ref).abs().max():.3g}, argmax-agree={(y.argmax(1)==ref.argmax(1)).float().mean():.5f}")
    print(line, flush=True)
