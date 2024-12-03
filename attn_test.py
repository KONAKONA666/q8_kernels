import torch
import math
from q8_matmul.flash_attention._C import flash_attention
from q8_matmul.ops._C import fast_hadamard_transform

l = 12
q = torch.load(f"../LTXVideo/acts/attn/q-{l}.pt")[:, :, :].cuda().contiguous().half()
k = torch.load(f"../LTXVideo/acts/attn/k-{l}.pt")[:, :, :].cuda().contiguous().half()
v = torch.load(f"../LTXVideo/acts/attn/v-{l}.pt")[:, :, :].cuda().contiguous().half()

head_dim = q.shape[-1]
sm_scale = 1/math.sqrt(head_dim)
sm_scale_fp8 = sm_scale

q_fp8 = q.to(torch.float8_e4m3fn)
k_fp8 = k.to(torch.float8_e4m3fn)
v_fp8 = v.transpose(2, 3).contiguous().to(torch.float8_e4m3fn)

q_hadamard = fast_hadamard_transform(q_fp8, 1/math.sqrt(head_dim)).to(torch.float8_e4m3fn)
k_hadamard = fast_hadamard_transform(k_fp8, 1/math.sqrt(head_dim)).to(torch.float8_e4m3fn)

torch.cuda.synchronize()
s = torch.cuda.Event(True)
e = torch.cuda.Event(True)
v_tokens = v.shape[-2]
v_tokens_pad = ((v_tokens + 15)//16)*16 - v_tokens
s.record()
v_fp8_padded = torch.nn.functional.pad(v_fp8, (0, v_tokens_pad))
e.record()
torch.cuda.synchronize()
print(s.elapsed_time(e))

TFLOPS_PER_ATTN = 4*q.shape[0]*q.shape[1]*q.shape[2]*q.shape[2]*q.shape[3]
int8_tflops = []
fp16_tflops = []

# for _ in range(10):
#     o = flash_attention_int8(q_quant, k_quant, v_quant, q_scales, k_scales, v_scales)
# flash_attention_int8_4stages
torch.cuda.synchronize()
N_ROUNDS = 10
N_OUTER_ROUNDS = 1
sm_scale_fp8 =sm_scale*1.44269504
for _ in range(5):
    o = flash_attention(q_fp8, k_fp8, v_fp8_padded, sm_scale, None)
torch.cuda.synchronize()

for _ in range(N_OUTER_ROUNDS):
    start_events = [ torch.cuda.Event(True) for _ in range(N_ROUNDS)]
    end_events = [ torch.cuda.Event(True) for _ in range(N_ROUNDS)]
    for i in range(N_ROUNDS):
        start_events[i].record()
        v_fp8_padded = torch.nn.functional.pad(v_fp8, (0, v_tokens_pad))
        o = flash_attention(q_fp8, k_fp8, v_fp8_padded, sm_scale, None)
        end_events[i].record()
    torch.cuda.synchronize()
    elapsed_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    int8_tflops.append((TFLOPS_PER_ATTN * 1e-12)/(min(elapsed_times) * 1e-3))
print(max(int8_tflops))


# for _ in range(10):
#     o = flash_attention_int8(q_quant, k_quant, v_quant, q_scales, k_scales, v_scales)
# flash_attention_int8_4stages
torch.cuda.synchronize()
N_ROUNDS = 10
N_OUTER_ROUNDS = 1
for _ in range(5):
    o_half = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=sm_scale)

torch.cuda.synchronize()

for _ in range(N_OUTER_ROUNDS):
    start_events = [ torch.cuda.Event(True) for _ in range(N_ROUNDS)]
    end_events = [ torch.cuda.Event(True) for _ in range(N_ROUNDS)]
    for i in range(N_ROUNDS):
        start_events[i].record()
        
        o_half = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=sm_scale)

        end_events[i].record()
    torch.cuda.synchronize()
    elapsed_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    fp16_tflops.append((TFLOPS_PER_ATTN * 1e-12)/(min(elapsed_times) * 1e-3))
print(max(fp16_tflops))

batch_mask = torch.tensor([32, 32], dtype=torch.int).cuda()

o_fp8 = flash_attention(q_fp8, k_fp8, v_fp8_padded, sm_scale, None)
o_fp8_h = flash_attention(q_hadamard, k_hadamard, v_fp8_padded, sm_scale, None)
o_half = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=sm_scale)
o_half_ref = torch.nn.functional.scaled_dot_product_attention(q.to(torch.float8_e4m3fn).half(), k.to(torch.float8_e4m3fn).half(), v.to(torch.float8_e4m3fn).half(), scale=sm_scale).to(torch.float8_e4m3fn)



def diff_max(a, b):
    return (a.float() - b.float()).abs().max()

def diff_quantiles(a, b):
    return torch.quantile((a.float() - b.float()).abs()[1, :, :], torch.tensor([0.25, 0.5, 0.75, 0.9, 0.99, 1.0]).cuda())
def diff_rms(a, b):
    return torch.sqrt(((a.float() - b.float()).square().sum()/a.numel()))


def cos_sim(a, b):
    a = a.float()
    b = b.float()
    a_len = a.norm(dim=-1, p=2)
    b_len = b.norm(dim=-1, p=2)
    dot_prod = (a * b).sum(dim=-1)
    return dot_prod/(a_len*b_len)


diff_fp8 = diff_max(o_fp8, o_half)
diff_h = diff_max(o_fp8_h, o_half)
diff_ideal = diff_max(o_half, o_half.to(torch.float8_e4m3fn))

print(diff_rms(o_half, o_fp8))
print(diff_rms(o_half, o_fp8_h))
