"""Fixed-size patch micro-benchmark of execution backends for one TotalSegmentator part model.

eager(autocast fp16 = original) | torch.compile default / max-autotune | ORT CUDA EP (fp16 onnx)
| TensorRT fp32 / fp16. Forward only, (B,1,128,128,128), warm-up then timed with cuda sync.
Build/compile times reported separately. Agreement vs eager: max|diff| and argmax agreement.

python benchmarks/microbench.py --weights <results dir> --out <json>
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "fastsegmentator")); sys.path.insert(0, str(ROOT / "flashseg")); sys.path.insert(0, str(ROOT / "benchmarks"))
import totalseg_infer as ti  # noqa: E402


def timeit(fn, x, n):
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n):
        fn(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--task_id", type=int, default=291)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--skip", nargs="*", default=[])
    a = ap.parse_args()
    dev = torch.device("cuda")
    cfg = ti.replace(ti.TASK_CONFIGS["total"], task_ids=(a.task_id,))
    p = ti.build_predictors(cfg, Path(a.weights), dev, 0.8)[a.task_id]
    net = p.network.eval()
    in_shape = (1, *p.configuration_manager.patch_size)
    res = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "backends": {}}

    def eager(x):
        with torch.no_grad(), torch.autocast("cuda"):
            return net(x)

    backends = {"eager_autocast": (eager, 0.0)}

    for mode in ("default", "max-autotune"):
        name = f"compile_{mode}"
        if name in a.skip:
            continue
        try:
            cnet = torch.compile(net, mode=mode, dynamic=False)

            def f(x, cnet=cnet):
                with torch.no_grad(), torch.autocast("cuda"):
                    return cnet(x)
            t0 = time.perf_counter()
            for B in a.batches:          # compile cost for every static batch shape
                f(torch.randn(B, *in_shape, device=dev)); torch.cuda.synchronize()
            backends[name] = (f, time.perf_counter() - t0)
        except Exception as e:
            res["backends"][name] = {"error": f"{type(e).__name__}: {e}"[:300]}

    if "trt" not in a.skip:
        try:
            from trt_backend import get_trt_module
            for prec in ("fp32", "fp16"):
                m, tm = get_trt_module(net, p.list_of_parameters[-1], in_shape, prec,
                                       max_batch=max(a.batches), opt_batch=max(a.batches))
                backends[f"trt_{prec}"] = (m, tm["onnx_export_s"] + tm["engine_build_s"])
                res.setdefault("trt_timing", {})[prec] = tm
        except Exception as e:
            res["backends"]["trt"] = {"error": f"{type(e).__name__}: {e}"[:500]}

    if "ort" not in a.skip:
        try:
            import glob
            import onnxruntime as ort
            onnx = sorted(glob.glob(str(ROOT / "trt_engines" / "onnx_*_fp16" / "model.onnx")))[-1]
            t0 = time.perf_counter()
            sess = ort.InferenceSession(onnx, providers=[("CUDAExecutionProvider",
                                                          {"cudnn_conv_algo_search": "EXHAUSTIVE"})])

            nh = p.label_manager.num_segmentation_heads

            def f(x, sess=sess):
                import numpy as np
                x = x.half().contiguous()
                y = torch.empty((x.shape[0], nh, *x.shape[2:]), dtype=torch.half, device=x.device)
                io = sess.io_binding()
                io.bind_input("x", "cuda", 0, np.float16, tuple(x.shape), x.data_ptr())
                io.bind_output("y", "cuda", 0, np.float16, tuple(y.shape), y.data_ptr())
                torch.cuda.synchronize()
                sess.run_with_iobinding(io)
                return y
            f(torch.randn(1, *in_shape, device=dev))
            backends["ort_cuda_fp16"] = (f, time.perf_counter() - t0)
        except Exception as e:
            res["backends"]["ort_cuda_fp16"] = {"error": f"{type(e).__name__}: {e}"[:300]}

    for B in a.batches:
        x = torch.randn(B, *in_shape, device=dev)
        ref = eager(x).float()
        for name, (fn, build_s) in backends.items():
            try:
                for _ in range(3):
                    fn(x)
                torch.cuda.reset_peak_memory_stats()
                t = timeit(fn, x, a.iters)
                y = fn(x).float()
                r = res["backends"].setdefault(name, {"build_or_compile_s": build_s})
                r[f"B{B}"] = {"ms_per_patch": t * 1e3 / B, "ms_per_batch": t * 1e3,
                              "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
                              "max_abs_diff": float((y - ref).abs().max()),
                              "argmax_agree": float((y.argmax(1) == ref.argmax(1)).float().mean())}
                print(name, B, r[f"B{B}"], flush=True)
            except Exception as e:
                res["backends"].setdefault(name, {})[f"B{B}"] = {"error": f"{type(e).__name__}: {e}"[:300]}
                print(name, B, "ERROR", e, flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
