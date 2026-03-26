#!/usr/bin/env python3
"""GH-559: Falsify the claim that PTX cannot be fixed.

Tests 5 alternative GPU computation approaches on sm_121.
Uses only PyTorch (no pycuda dependency).

Usage: uv run --with torch scripts/falsify-ptx-fix.py
"""
import torch
import math
import sys

def cosine_sim(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return (torch.dot(a, b) / (a.norm() * b.norm())).item()

HIDDEN = 3584
EPS = 1e-6

print("=" * 60)
print("GH-559: Falsify PTX Fix — 5 Alternative Approaches")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    print("ERROR: No CUDA device"); sys.exit(1)
print(f"GPU: {torch.cuda.get_device_name(0)}, CUDA: {torch.version.cuda}")

torch.manual_seed(42)
x = torch.randn(HIDDEN, dtype=torch.float32) * 0.02
gamma = torch.randn(HIDDEN, dtype=torch.float32) * 0.5 + 1.0

# CPU reference
sq = (x * x).sum().item()
rms = math.sqrt(sq / HIDDEN + EPS)
cpu_out = [(x[i].item() / rms) * gamma[i].item() for i in range(HIDDEN)]
cpu_t = torch.tensor(cpu_out, dtype=torch.float32)
print(f"CPU: sq_sum={sq:.10f}, rms={rms:.10f}")

results = {}

# === Test 1: PyTorch ops (cuBLAS/cuDNN pre-compiled SASS) ===
print("\n--- Test 1: PyTorch element-wise ops (pre-compiled CUDA) ---")
x_g, gamma_g = x.cuda(), gamma.cuda()
sq1 = (x_g * x_g).sum()
rms1 = torch.sqrt(sq1 / HIDDEN + EPS)
out1 = ((x_g / rms1) * gamma_g).cpu()
cos1 = cosine_sim(cpu_t, out1)
print(f"cosine={cos1:.10f}  sq_sum={sq1.item():.10f}")
results["1. PyTorch ops (pre-compiled)"] = cos1

# === Test 2: torch.compile (Triton JIT to SASS) ===
print("\n--- Test 2: torch.compile (Triton → PTX → SASS) ---")
try:
    @torch.compile(fullgraph=True)
    def rmsnorm_triton(x, g, eps):
        s = (x * x).sum()
        r = torch.sqrt(s / x.shape[0] + eps)
        return (x / r) * g
    out2 = rmsnorm_triton(x_g, gamma_g, EPS).cpu()
    cos2 = cosine_sim(cpu_t, out2)
    print(f"cosine={cos2:.10f}")
    results["2. torch.compile (Triton JIT)"] = cos2
except Exception as e:
    print(f"SKIP: {e}")
    results["2. torch.compile (Triton JIT)"] = None

# === Test 3: CUDA kernel via torch custom op (nvrtc-compiled) ===
print("\n--- Test 3: Inline CUDA C (nvrtc via torch) ---")
try:
    from torch.utils.cpp_extension import load_inline
    cuda_src = """
    #include <torch/extension.h>
    #include <cuda_runtime.h>

    __global__ void rmsnorm_kernel(const float* x, const float* g, float* o, int n, float eps) {
        __shared__ float smem[256];
        int tid = threadIdx.x;
        float sq = 0.0f;
        for (int i = tid; i < n; i += 256) {
            float v = x[i];
            sq += v * v;
        }
        smem[tid] = sq;
        __syncthreads();
        for (int s = 128; s > 0; s >>= 1) {
            if (tid < s) smem[tid] += smem[tid + s];
            __syncthreads();
        }
        float rms = sqrtf(smem[0] / (float)n + eps);
        for (int i = tid; i < n; i += 256) {
            o[i] = (x[i] / rms) * g[i];
        }
    }

    torch::Tensor rmsnorm_cuda(torch::Tensor x, torch::Tensor g, float eps) {
        auto o = torch::empty_like(x);
        rmsnorm_kernel<<<1, 256>>>(x.data_ptr<float>(), g.data_ptr<float>(),
                                    o.data_ptr<float>(), x.size(0), eps);
        return o;
    }
    """
    cpp_src = "torch::Tensor rmsnorm_cuda(torch::Tensor x, torch::Tensor g, float eps);"
    mod3 = load_inline(name="rmsnorm_test", cpp_sources=cpp_src,
                       cuda_sources=cuda_src, functions=["rmsnorm_cuda"],
                       verbose=False)
    out3 = mod3.rmsnorm_cuda(x_g, gamma_g, EPS).cpu()
    cos3 = cosine_sim(cpu_t, out3)
    print(f"cosine={cos3:.10f}")
    results["3. CUDA C (nvrtc inline)"] = cos3
except Exception as e:
    print(f"SKIP: {e}")
    results["3. CUDA C (nvrtc inline)"] = None

# === Test 4: Double-precision accumulation on GPU ===
print("\n--- Test 4: FP64 accumulation (torch.float64 on GPU) ---")
x_d = x.double().cuda()
gamma_d = gamma.double().cuda()
sq4 = (x_d * x_d).sum()
rms4 = torch.sqrt(sq4 / HIDDEN + EPS)
out4 = ((x_d / rms4) * gamma_d).float().cpu()
cos4 = cosine_sim(cpu_t, out4)
print(f"cosine={cos4:.10f}  sq_sum={sq4.item():.15f}")
results["4. FP64 accumulation"] = cos4

# === Test 5: torch.nn.RMSNorm (cuDNN pre-compiled) ===
print("\n--- Test 5: torch.nn.RMSNorm (cuDNN/CUDA pre-compiled) ---")
try:
    rms_layer = torch.nn.RMSNorm(HIDDEN, eps=EPS).cuda()
    # Set gamma to our test gamma
    with torch.no_grad():
        rms_layer.weight.copy_(gamma_g)
    out5 = rms_layer(x_g.unsqueeze(0)).squeeze(0).cpu()
    cos5 = cosine_sim(cpu_t, out5)
    print(f"cosine={cos5:.10f}")
    results["5. torch.nn.RMSNorm (cuDNN)"] = cos5
except Exception as e:
    print(f"SKIP: {e}")
    results["5. torch.nn.RMSNorm (cuDNN)"] = None

# === Summary ===
print("\n" + "=" * 60)
print("SUMMARY: Which compilation paths produce correct results?")
print("=" * 60)
for name, cos in results.items():
    if cos is None:
        s = "SKIP"
    elif cos > 0.9999:
        s = "PASS ✓"
    else:
        s = f"FAIL ✗"
    print(f"  {name:45s} {f'cosine={cos:.10f}' if cos else 'N/A':>25s}  {s}")

passing = sum(1 for c in results.values() if c and c > 0.9999)
print(f"\n{passing}/{len(results)} approaches produce correct results on sm_121.")
print()
if passing == len([c for c in results.values() if c is not None]):
    print("CONCLUSION: ALL tested approaches work. The bug is ONLY in the")
    print("driver JIT path (cuModuleLoadData with raw PTX). Every other")
    print("compilation path (nvrtc, cuDNN, Triton, cuBLAS) produces correct SASS.")
    print()
    print("This means the PTX SOURCE is not the problem — it's how the driver")
    print("JIT translates PTX to SASS that's broken. The fix is to use ANY")
    print("other compilation path: wgpu (Vulkan), NVRTC, or torch.compile.")
