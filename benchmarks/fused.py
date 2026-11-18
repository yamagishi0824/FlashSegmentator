"""Fuse k PlainConvUNets of identical topology (TotalSegmentator `total` parts) into one network
built from grouped convolutions (PERF_LOG §8). Mathematically the same computation as running the
k networks separately; one kernel launch per layer instead of k.

input : (B, k, D, H, W)  -- channel m = patch normalised with model m's own intensity statistics
output: list of k tensors (B, nh_m, D, H, W)

Layout: every activation is (B, k*C, ...) with model m occupying channels [m*C, (m+1)*C).
  conv / transposed conv: groups=k, weights concatenated along the output dim
  InstanceNorm: per channel -> affine params concatenated
  decoder skip concat: interleaved per model (view (B,k,C,...) and cat on dim 2)
  seg head: class counts differ per model -> zero-padded to max, sliced on output
Only the models needed for the request are fused (e.g. --roi_subset may need 1–2 of the 5).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _leaf_blocks(module):
    """[(conv, norm, nonlin), ...] in forward order for a StackedConvBlocks-like module."""
    out = []
    for m in module.modules():
        if hasattr(m, "conv") and hasattr(m, "norm") and isinstance(m.conv, nn.Conv3d):
            out.append((m.conv, m.norm, m.nonlin))
    return out


def _fuse_conv(convs, k, shared_input=False):
    c0 = convs[0]
    groups = 1 if shared_input else k
    in_ch = c0.in_channels if shared_input else c0.in_channels * k
    f = nn.Conv3d(in_ch, c0.out_channels * k, c0.kernel_size, c0.stride, c0.padding,
                  bias=c0.bias is not None, groups=groups)
    with torch.no_grad():
        f.weight.copy_(torch.cat([c.weight for c in convs], 0))
        if c0.bias is not None:
            f.bias.copy_(torch.cat([c.bias for c in convs], 0))
    return f


def _fuse_norm(norms):
    n0 = norms[0]
    f = nn.InstanceNorm3d(n0.num_features * len(norms), eps=n0.eps, affine=n0.affine,
                          track_running_stats=False)
    if n0.affine:
        with torch.no_grad():
            f.weight.copy_(torch.cat([n.weight for n in norms]))
            f.bias.copy_(torch.cat([n.bias for n in norms]))
    return f


def _fuse_tconv(tcs, k):
    t0 = tcs[0]
    f = nn.ConvTranspose3d(t0.in_channels * k, t0.out_channels * k, t0.kernel_size, t0.stride,
                           bias=t0.bias is not None, groups=k)
    with torch.no_grad():
        f.weight.copy_(torch.cat([t.weight for t in tcs], 0))      # (k*in, out, ...)
        if t0.bias is not None:
            f.bias.copy_(torch.cat([t.bias for t in tcs], 0))
    return f


class _Block(nn.Module):
    def __init__(self, convs, norms, nonlin, k, first=False):
        super().__init__()
        self.conv = _fuse_conv(convs, k, shared_input=False)
        self.norm = _fuse_norm(norms)
        self.nonlin = nonlin

    def forward(self, x):
        return self.nonlin(self.norm(self.conv(x)))


class FusedUNet(nn.Module):
    def __init__(self, nets):
        super().__init__()
        self.k = k = len(nets)
        nonlin = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        # encoder
        self.enc = nn.ModuleList()
        for s in range(len(nets[0].encoder.stages)):
            blocks = [_leaf_blocks(n.encoder.stages[s]) for n in nets]
            self.enc.append(nn.Sequential(*[
                _Block([b[j][0] for b in blocks], [b[j][1] for b in blocks], nonlin, k)
                for j in range(len(blocks[0]))]))
        # decoder
        d0 = nets[0].decoder
        self.tconv = nn.ModuleList([_fuse_tconv([n.decoder.transpconvs[s] for n in nets], k)
                                    for s in range(len(d0.stages))])
        self.dec = nn.ModuleList()
        for s in range(len(d0.stages)):
            blocks = [_leaf_blocks(n.decoder.stages[s]) for n in nets]
            self.dec.append(nn.Sequential(*[
                _Block([b[j][0] for b in blocks], [b[j][1] for b in blocks], nonlin, k)
                for j in range(len(blocks[0]))]))
        # seg head (1x1x1), zero-pad class dim to max
        heads = [n.decoder.seg_layers[-1] for n in nets]
        self.nh = [h.out_channels for h in heads]
        nmax = max(self.nh); cin = heads[0].in_channels
        self.head = nn.Conv3d(cin * k, nmax * k, 1, bias=True, groups=k)
        with torch.no_grad():
            self.head.weight.zero_(); self.head.bias.zero_()
            for m, h in enumerate(heads):
                self.head.weight[m * nmax: m * nmax + h.out_channels] = h.weight
                self.head.bias[m * nmax: m * nmax + h.out_channels] = h.bias
        self.nmax = nmax

    def _cat_skip(self, x, skip):
        b = x.shape[0]; sp = x.shape[2:]
        x = x.view(b, self.k, -1, *sp); skip = skip.view(b, self.k, -1, *sp)
        return torch.cat((x, skip), 2).reshape(b, -1, *sp)

    def forward(self, x):
        skips = []
        for st in self.enc:
            x = st(x); skips.append(x)
        x = skips[-1]
        for s in range(len(self.dec)):
            x = self.tconv[s](x)
            x = self._cat_skip(x, skips[-(s + 2)])
            x = self.dec[s](x)
        y = self.head(x)
        b, sp = y.shape[0], y.shape[2:]
        y = y.view(b, self.k, self.nmax, *sp)
        return [y[:, m, :self.nh[m]] for m in range(self.k)]
