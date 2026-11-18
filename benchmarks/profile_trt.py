"""Per-layer TensorRT profile of a cached engine: python benchmarks/profile_trt.py <engine.plan> [batch]"""
import sys, collections, torch, tensorrt as trt
sys.path.insert(0, __file__.rsplit('/', 1)[0]); sys.path.insert(0, __file__.rsplit('/', 2)[0] + '/flashseg')
from trt_backend import TRTModule
class P(trt.IProfiler):
    def __init__(s): super().__init__(); s.t = collections.defaultdict(float)
    def report_layer_time(s, name, ms): s.t[name] += ms
m = TRTModule(sys.argv[1]); B = int(sys.argv[2]) if len(sys.argv) > 2 else 1
x = torch.randn(B, 1, 128, 128, 128, device='cuda')
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    m(x); torch.cuda.synchronize()
    prof = P(); m.ctx.profiler = prof
    m(x); torch.cuda.synchronize()
tot = sum(prof.t.values()); print(f'total {tot:.1f} ms  layers={len(prof.t)}')
for k, v in sorted(prof.t.items(), key=lambda kv: -kv[1])[:12]: print(f'{v:8.1f} ms  {k[:150]}')
insp = m.engine.create_engine_inspector()
info = insp.get_engine_information(trt.LayerInformationFormat.JSON)
import json; L = json.loads(info)['Layers']; print('n layers', len(L))
for l in L[:200]:
    if isinstance(l, dict) and l.get('LayerType') in ('CaskConvolution','Convolution','Myelin','gemm') or 'conv' in str(l)[:300].lower():
        print(str({k: l.get(k) for k in ('Name','LayerType','TacticName','Tactic')})[:250]); 
