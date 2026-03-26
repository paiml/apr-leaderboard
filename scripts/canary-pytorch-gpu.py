#!/usr/bin/env python3
"""GH-559 Canary: Prove GPU hardware works with PyTorch (pre-compiled CUDA kernels).

If PyTorch GPU/CPU match but our PTX kernels don't, the bug is in JIT compilation.
If PyTorch also fails, the bug is in the hardware/driver.

Usage: uv run --with torch scripts/canary-pytorch-gpu.py
"""

import torch
import sys
import math

def cosine_sim(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return (torch.dot(a, b) / (a.norm() * b.norm())).item()

def rms_norm(x, gamma, eps=1e-6):
    """Manual RMSNorm matching our CUDA kernel formula."""
    sq_sum = (x * x).sum()
    rms = torch.sqrt(sq_sum / x.shape[0] + eps)
    return (x / rms) * gamma

print("=" * 60)
print("GH-559 PyTorch GPU Canary")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type != "cuda":
    print("ERROR: No CUDA device available")
    sys.exit(1)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA: {torch.version.cuda}")
print(f"PyTorch: {torch.__version__}")
print()

# Test 1: RMSNorm on known input (matches our model dimensions)
print("=== Test 1: RMSNorm (hidden_dim=3584, eps=1e-6) ===")
torch.manual_seed(42)
hidden_dim = 3584
eps = 1e-6

# Create input matching our embedding scale (~0.01 values)
x = torch.randn(hidden_dim, dtype=torch.float32) * 0.02
gamma = torch.randn(hidden_dim, dtype=torch.float32) * 0.5 + 1.0

# CPU RMSNorm
cpu_out = rms_norm(x, gamma, eps)

# GPU RMSNorm
x_gpu = x.cuda()
gamma_gpu = gamma.cuda()
gpu_out = rms_norm(x_gpu, gamma_gpu, eps).cpu()

cos = cosine_sim(cpu_out, gpu_out)
max_diff = (cpu_out - gpu_out).abs().max().item()
print(f"CPU first 5: {cpu_out[:5].tolist()}")
print(f"GPU first 5: {gpu_out[:5].tolist()}")
print(f"Cosine: {cos:.10f}")
print(f"Max abs diff: {max_diff:.2e}")
print(f"Result: {'PASS' if cos > 0.9999 else 'FAIL'}")
print()

# Test 2: PyTorch nn.RMSNorm (uses cuDNN/compiled CUDA)
print("=== Test 2: torch.nn.RMSNorm ===")
try:
    rms_layer = torch.nn.RMSNorm(hidden_dim, eps=eps)
    with torch.no_grad():
        cpu_out2 = rms_layer(x.unsqueeze(0)).squeeze(0)
        rms_layer_gpu = rms_layer.cuda()
        gpu_out2 = rms_layer_gpu(x_gpu.unsqueeze(0)).squeeze(0).cpu()
    cos2 = cosine_sim(cpu_out2, gpu_out2)
    max_diff2 = (cpu_out2 - gpu_out2).abs().max().item()
    print(f"Cosine: {cos2:.10f}")
    print(f"Max abs diff: {max_diff2:.2e}")
    print(f"Result: {'PASS' if cos2 > 0.9999 else 'FAIL'}")
except Exception as e:
    print(f"SKIP: {e}")
print()

# Test 3: Linear layer (GEMV equivalent)
print("=== Test 3: Linear (3584→3584, Q=W×x) ===")
linear = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
with torch.no_grad():
    cpu_out3 = linear(x.unsqueeze(0)).squeeze(0)
    linear_gpu = linear.cuda()
    gpu_out3 = linear_gpu(x_gpu.unsqueeze(0)).squeeze(0).cpu()
cos3 = cosine_sim(cpu_out3, gpu_out3)
max_diff3 = (cpu_out3 - gpu_out3).abs().max().item()
print(f"Cosine: {cos3:.10f}")
print(f"Max abs diff: {max_diff3:.2e}")
print(f"Result: {'PASS' if cos3 > 0.999 else 'FAIL'}")
print()

# Test 4: Full transformer layer
print("=== Test 4: TransformerEncoderLayer (d=3584, nhead=28) ===")
try:
    layer = torch.nn.TransformerEncoderLayer(
        d_model=hidden_dim, nhead=28, dim_feedforward=hidden_dim*4,
        dropout=0.0, batch_first=True, norm_first=True
    )
    layer.eval()
    inp = x.unsqueeze(0).unsqueeze(0)  # (1, 1, 3584)
    with torch.no_grad():
        cpu_out4 = layer(inp).squeeze()
        layer_gpu = layer.cuda()
        gpu_out4 = layer_gpu(inp.cuda()).squeeze().cpu()
    cos4 = cosine_sim(cpu_out4, gpu_out4)
    max_diff4 = (cpu_out4 - gpu_out4).abs().max().item()
    print(f"Cosine: {cos4:.10f}")
    print(f"Max abs diff: {max_diff4:.2e}")
    print(f"Result: {'PASS' if cos4 > 0.999 else 'FAIL'}")
except Exception as e:
    print(f"SKIP: {e}")
print()

# Test 5: Warp shuffle equivalent (sum reduction)
print("=== Test 5: Sum reduction (3584 elements) ===")
cpu_sum = x.sum().item()
gpu_sum = x_gpu.sum().cpu().item()
sum_diff = abs(cpu_sum - gpu_sum)
sum_rel = sum_diff / max(abs(cpu_sum), 1e-10) * 100
print(f"CPU sum: {cpu_sum:.10f}")
print(f"GPU sum: {gpu_sum:.10f}")
print(f"Diff: {sum_diff:.2e} ({sum_rel:.6f}%)")
print(f"Result: {'PASS' if sum_rel < 0.01 else 'FAIL'}")
print()

# Summary
print("=" * 60)
results = [
    ("RMSNorm manual", cos > 0.9999),
    ("Linear 3584→3584", cos3 > 0.999),
    ("Sum reduction", sum_rel < 0.01),
]
passed = sum(1 for _, p in results if p)
total = len(results)
for name, ok in results:
    print(f"  {name}: {'PASS' if ok else 'FAIL'}")
print(f"\n{passed}/{total} passed")

if passed == total:
    print("\nCONCLUSION: GPU hardware works correctly with PyTorch.")
    print("The bug is in our JIT'd PTX compilation (trueno-gpu), NOT the hardware.")
else:
    print("\nCONCLUSION: GPU hardware has issues even with PyTorch.")
    print("This is a driver/hardware bug, not specific to our PTX.")
