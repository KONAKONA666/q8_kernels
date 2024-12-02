import os
import torch
import math

from q8_gemm import q8_mm
from fast_hadamard_transform import hadamard_transform

from safetensors.torch import load_file

import torch.nn.functional as F

def diff_max(a, b):
    return (a.float() - b.float()).abs().max()

def diff_mean(a, b):
    return (a.float() - b.float()).abs().mean()


def diff_quantiles(a, b):
    if a.ndim > 2:
        return torch.quantile((a.float() - b.float()).abs()[1, :2048, :], torch.tensor([0.25, 0.5, 0.75, 0.9, 0.99, 1.0]).cuda())
    else:
        return torch.quantile((a.float() - b.float()).abs()[:2048, :], torch.tensor([0.25, 0.5, 0.75, 0.9, 0.99, 1.0]).cuda())


def gelu(gate, approximate="tanh"):
    return F.gelu(gate.to(dtype=torch.float32), approximate=approximate).to(dtype=gate.dtype)



def hadamard_quant(x):
    k = x.shape[-1]
    x_hadamard = hadamard_transform(x, scale=1/math.sqrt(k))
    x_abs_max_hadamard = x_hadamard.float().abs().max(-1, False).values
    x_scale_hadamard = x_abs_max_hadamard/127.0
    x_q8_hadamard = (x_hadamard.float() / x_scale_hadamard[..., None]).round().to(torch.int8)
    return x_q8_hadamard, x_scale_hadamard

def quant(x):
    x_abs_max = x.float().abs().max(-1, False).values
    x_scale = x_abs_max/127.0
    x_q8 =  (x.float() / x_scale[..., None]).round().to(torch.int8)
    return x_q8, x_scale


l_idx = 3
x = torch.load(f"/data/LTXVideo/acts/ffn/hs-{l_idx}.pt", map_location="cuda")[:, :, :]
model_weights = load_file("/data/ltx_weights/unet/unet_diffusion_pytorch_model.safetensors", device="cpu")
w = model_weights[f"transformer_blocks.{l_idx}.ff.net.0.proj.weight"].cuda()

k = x.shape[-1]
x_hadamard = hadamard_transform(x.to(torch.float8_e4m3fn), scale=1/math.sqrt(k))
w_hadamard = hadamard_transform(w.to(torch.float8_e4m3fn), scale=1/math.sqrt(k))

x_quant_h, x_scales_h = hadamard_quant(x.to(torch.float8_e4m3fn))
w_quant_h, w_scales_h = hadamard_quant(w.to(torch.float8_e4m3fn))

x_quant, x_scales = quant(x)
w_quant, w_scales = quant(w)

o_q8_h = q8_mm(x_quant_h, w_quant_h, x_scales_h, w_scales_h, True)
o_q8 = q8_mm(x_quant, w_quant, x_scales, w_scales, True)

o_q8_torch_h = gelu(((x_scales_h[..., None] * w_scales_h[None, None, :]) * torch.matmul(x_quant_h.float(), w_quant_h.float().t()))).to(torch.float8_e4m3fn)
o_q8_torch = gelu(((x_scales[..., None] * w_scales[None, None, :]) * torch.matmul(x_quant.float(), w_quant.float().t()))).to(torch.float8_e4m3fn)


o_orig = gelu(torch.matmul(x.to(torch.float8_e4m3fn).half(), w.to(torch.float8_e4m3fn).half().t()))# nn.functional.linear(x, w, bias=None)


diff_q8_h = diff_max(o_q8_h, o_orig)
diff_q8 = diff_max(o_q8, o_orig)
diff_q8_torch_h = diff_max(o_q8_torch_h, o_orig)
diff_q8_torch = diff_max(o_q8_torch, o_orig)

o_fp8_orig = gelu(torch._scaled_mm(x[1].to(torch.float8_e4m3fn),  w.to(torch.float8_e4m3fn).contiguous().t(), scale_a=torch.tensor([1.0]).cuda(), scale_b=torch.tensor([1.0]).cuda()))

print("DIFF Hadamard: ", diff_q8_h)
print("DIFF no Hadamard: ", diff_q8)

print("DIFF Hadamard Torch: ", diff_q8_torch_h)
print("DIFF no Hadamard Torch: ", diff_q8_torch)

print("torch q8 vs cute q8: ", diff_max(o_q8_torch_h, o_q8_h))
print("torch fp8 vs fp16: ", diff_max(o_fp8_orig, o_orig[1]))

print("diff torch fp8 vs q8: ", diff_max(o_fp8_orig, o_q8_h[1]))


torch.cuda.synchronize()
N_ROUNDS = 10
N_OUTER_ROUNDS = 1

batch = x.shape[0]
m = x.shape[1]
n = w.shape[0]
k = x.shape[-1]
int8_tflops = []
TFLOPS = 2 * batch * m *n *k 

for _ in range(5):
    o_q8_h = q8_mm(x_quant_h, w_quant_h, x_scales_h, w_scales_h, True)

torch.cuda.synchronize()

for _ in range(N_OUTER_ROUNDS):
    start_events = [ torch.cuda.Event(True) for _ in range(N_ROUNDS)]
    end_events = [ torch.cuda.Event(True) for _ in range(N_ROUNDS)]
    for i in range(N_ROUNDS):
        start_events[i].record()
        o_q8_h = q8_mm(x_quant_h, w_quant_h, x_scales_h, w_scales_h, True)
        end_events[i].record()
    torch.cuda.synchronize()
    elapsed_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    int8_tflops.append((TFLOPS * 1e-12)/(min(elapsed_times) * 1e-3))

print("TFLOPS: ", max(int8_tflops))


torch.cuda.synchronize()
N_ROUNDS = 10
N_OUTER_ROUNDS = 1
fp8_tflops = []

_w = w_hadamard.contiguous().t()
_scale = torch.tensor([1.0]).cuda()
for _ in range(5):
    o_fp8_orig = torch._scaled_mm(x_hadamard[0],  _w, scale_a=_scale, scale_b=_scale)

torch.cuda.synchronize()
for _ in range(N_OUTER_ROUNDS):
    start_events = [ torch.cuda.Event(True) for _ in range(N_ROUNDS)]
    end_events = [ torch.cuda.Event(True) for _ in range(N_ROUNDS)]
    for i in range(N_ROUNDS):
        start_events[i].record()
        o_fp8_orig = gelu(torch._scaled_mm(x_hadamard[0],  _w, scale_a=torch.tensor([1.0]).cuda(), scale_b=torch.tensor([1.0]).cuda()))
        end_events[i].record()
    torch.cuda.synchronize()
    elapsed_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    fp8_tflops.append((TFLOPS/2 * 1e-12)/(min(elapsed_times) * 1e-3))

print("INT8/FP8 TFLOPS: ", max(int8_tflops)/max(fp8_tflops))