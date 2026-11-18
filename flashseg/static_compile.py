"""Static-shape torch.compile for the sliding-window network (PERF_LOG §6).

FastSegmentator's existing `nnUNet_compile=1` calls torch.compile(network) with defaults; with
batched / pipelined inference the batch dimension varies (tail batches), torch.compile then
recompiles the graph as dynamic-shape and the speed-up is lost (GH200: pipeline_b4 1670 -> 1287 CT/h).

Here: torch.compile(dynamic=False) and every call is padded to exactly `batch_size` patches
(zero padding, extra outputs dropped), so exactly one static graph per model is compiled.
Padding rows are independent samples (InstanceNorm statistics are per sample), so real outputs
are unaffected.
"""
import torch
import torch.nn as nn


# PERF_LOG §12: each (model, batch size) is a separate graph for the same forward code object; with 5 models
# and several batch sizes in one process the default dynamo cache limit (8) is exceeded and torch silently
# falls back to eager. Raise it.
torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
if hasattr(torch._dynamo.config, "accumulated_cache_size_limit"):
    torch._dynamo.config.accumulated_cache_size_limit = max(torch._dynamo.config.accumulated_cache_size_limit, 512)


class StaticBatchCompiled(nn.Module):
    def __init__(self, net: nn.Module, batch_size: int, mode: str = "default"):
        super().__init__()
        self.net = net
        self.batch_size = batch_size
        self.compiled = torch.compile(net, dynamic=False, mode=mode)

    def forward(self, x):
        n = x.shape[0]
        if n == self.batch_size:
            return self.compiled(x)
        if n > self.batch_size:
            return torch.cat([self(c) for c in x.split(self.batch_size)])
        pad = x.new_zeros((self.batch_size - n, *x.shape[1:]))
        return self.compiled(torch.cat([x, pad]))[:n]


def apply(predictors: dict, batch_size: int, mode: str = "default"):
    for p in predictors.values():
        p.network = StaticBatchCompiled(p.network.eval(), batch_size, mode)
