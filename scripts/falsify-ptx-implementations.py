#!/usr/bin/env python3
"""GH-559: Falsify by testing 5 different PTX implementations through cuModuleLoadData.

Each variant computes RMSNorm using a DIFFERENT PTX algorithm, loaded through
the SAME cuModuleLoadData JIT path that our trueno-gpu uses. If ANY variant
produces correct results, the PTX CAN be fixed.

Usage: uv run --with torch scripts/falsify-ptx-implementations.py
"""
import torch
import ctypes
import ctypes.util
import math
import struct
import sys

HIDDEN = 3584
EPS = 1e-6
eps_hex = struct.unpack('I', struct.pack('f', EPS))[0]

def cosine_sim(a, b):
    a, b = torch.tensor(a).double(), torch.tensor(b).double()
    return (torch.dot(a, b) / (a.norm() * b.norm())).item()

# Load CUDA driver API
libcuda = ctypes.CDLL("libcuda.so.1")

def check(err, msg=""):
    if err != 0:
        raise RuntimeError(f"CUDA error {err}: {msg}")

# Init + get context from PyTorch
torch.cuda.init()
_ = torch.zeros(1, device="cuda")  # force context creation

def load_and_run_ptx(ptx_source, kernel_name, x_gpu, gamma_gpu, out_gpu, n, block_x=32):
    """Load PTX via cuModuleLoadData (same path as trueno-gpu), run kernel."""
    module = ctypes.c_void_p()
    ptx_bytes = ptx_source.encode('utf-8') + b'\0'
    err = libcuda.cuModuleLoadData(ctypes.byref(module), ptx_bytes)
    if err != 0:
        return None, f"cuModuleLoadData failed: {err}"

    func = ctypes.c_void_p()
    err = libcuda.cuModuleGetFunction(ctypes.byref(func), module,
                                       kernel_name.encode('utf-8'))
    if err != 0:
        return None, f"cuModuleGetFunction failed: {err}"

    # Launch kernel
    x_ptr = ctypes.c_uint64(x_gpu.data_ptr())
    g_ptr = ctypes.c_uint64(gamma_gpu.data_ptr())
    o_ptr = ctypes.c_uint64(out_gpu.data_ptr())

    args = (ctypes.c_void_p * 3)(
        ctypes.cast(ctypes.pointer(x_ptr), ctypes.c_void_p),
        ctypes.cast(ctypes.pointer(o_ptr), ctypes.c_void_p),
        ctypes.cast(ctypes.pointer(g_ptr), ctypes.c_void_p),
    )

    err = libcuda.cuLaunchKernel(
        func,
        1, 1, 1,        # grid
        block_x, 1, 1,  # block
        0, None,         # shared mem, stream
        args, None       # args, extras
    )
    if err != 0:
        return None, f"cuLaunchKernel failed: {err}"

    err = libcuda.cuCtxSynchronize()
    if err != 0:
        return None, f"cuCtxSynchronize failed: {err}"

    result = out_gpu.cpu().tolist()
    return result, None

# ---- Generate test data ----
torch.manual_seed(42)
x = torch.randn(HIDDEN, dtype=torch.float32) * 0.02
gamma = torch.randn(HIDDEN, dtype=torch.float32) * 0.5 + 1.0

sq = (x * x).sum().item()
rms = math.sqrt(sq / HIDDEN + EPS)
cpu_out = [(x[i].item() / rms) * gamma[i].item() for i in range(HIDDEN)]

x_gpu = x.cuda()
gamma_gpu = gamma.cuda()

print("=" * 70)
print("GH-559: 5 PTX Implementations via cuModuleLoadData (same JIT path)")
print("=" * 70)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Hidden: {HIDDEN}, CPU rms: {rms:.10f}")

results = {}

# ===== Variant 1: Our original PTX (32-thread warp shuffle) =====
ptx_v1 = f"""
.version 8.0
.target sm_90
.address_size 64
.visible .entry rmsnorm_v1(
    .param .u64 input_ptr, .param .u64 output_ptr, .param .u64 gamma_ptr
) {{
    .reg .u64 %rd<9>; .reg .u32 %r<7>; .reg .f32 %f<17>; .reg .pred %p<2>;
    mov.u32 %r0, %tid.x;
    ld.param.u64 %rd0, [input_ptr]; ld.param.u64 %rd1, [output_ptr]; ld.param.u64 %rd2, [gamma_ptr];
    mov.u32 %r1, {HIDDEN}; mov.u32 %r2, 4; mov.f32 %f0, 0F00000000; mov.u32 %r3, 0;
L1: add.u32 %r4, %r3, %r0; setp.lt.u32 %p0, %r4, %r1; @!%p0 bra L2;
    mul.wide.u32 %rd3, %r4, %r2; add.u64 %rd4, %rd0, %rd3; ld.global.f32 %f1, [%rd4];
    fma.rn.f32 %f0, %f1, %f1, %f0; add.u32 %r3, %r3, 32; bra L1;
L2: shfl.sync.down.b32 %f2, %f0, 16, 31, 0xFFFFFFFF; add.f32 %f0, %f0, %f2;
    shfl.sync.down.b32 %f3, %f0, 8, 31, 0xFFFFFFFF; add.f32 %f0, %f0, %f3;
    shfl.sync.down.b32 %f4, %f0, 4, 31, 0xFFFFFFFF; add.f32 %f0, %f0, %f4;
    shfl.sync.down.b32 %f5, %f0, 2, 31, 0xFFFFFFFF; add.f32 %f0, %f0, %f5;
    shfl.sync.down.b32 %f6, %f0, 1, 31, 0xFFFFFFFF; add.f32 %f0, %f0, %f6;
    shfl.sync.idx.b32 %f7, %f0, 0, 31, 0xFFFFFFFF;
    cvt.rn.f32.u32 %f8, %r1; div.rn.f32 %f9, %f7, %f8;
    mov.f32 %f10, 0F{eps_hex:08X}; add.f32 %f11, %f9, %f10; rsqrt.approx.f32 %f12, %f11;
    mov.u32 %r5, 0;
L3: add.u32 %r6, %r5, %r0; setp.lt.u32 %p1, %r6, %r1; @!%p1 bra L4;
    mul.wide.u32 %rd5, %r6, %r2; add.u64 %rd6, %rd0, %rd5; add.u64 %rd7, %rd2, %rd5; add.u64 %rd8, %rd1, %rd5;
    ld.global.f32 %f13, [%rd6]; ld.global.f32 %f14, [%rd7];
    mul.f32 %f15, %f13, %f12; mul.f32 %f16, %f15, %f14; st.global.f32 [%rd8], %f16;
    add.u32 %r5, %r5, 32; bra L3;
L4: ret;
}}
"""

print("\n--- V1: Original (32-thread, warp shuffle, .target sm_90) ---")
out_gpu = torch.zeros(HIDDEN, dtype=torch.float32, device="cuda")
result, err = load_and_run_ptx(ptx_v1, "rmsnorm_v1", x_gpu, gamma_gpu, out_gpu, HIDDEN, 32)
if err:
    print(f"ERROR: {err}")
    results["V1: Original (warp shuffle)"] = None
else:
    cos = cosine_sim(cpu_out, result)
    print(f"cosine={cos:.10f}")
    results["V1: Original (warp shuffle)"] = cos

# ===== Variant 2: Single-threaded (no parallelism at all) =====
ptx_v2 = f"""
.version 8.0
.target sm_90
.address_size 64
.visible .entry rmsnorm_v2(
    .param .u64 input_ptr, .param .u64 output_ptr, .param .u64 gamma_ptr
) {{
    .reg .u64 %rd<6>; .reg .u32 %r<4>; .reg .f32 %f<10>; .reg .pred %p<2>;
    mov.u32 %r0, %tid.x; setp.ne.u32 %p1, %r0, 0; @%p1 bra DONE;
    ld.param.u64 %rd0, [input_ptr]; ld.param.u64 %rd1, [output_ptr]; ld.param.u64 %rd2, [gamma_ptr];
    mov.u32 %r1, {HIDDEN}; mov.u32 %r2, 4; mov.f32 %f0, 0F00000000; mov.u32 %r3, 0;
SUM: setp.lt.u32 %p0, %r3, %r1; @!%p0 bra NORM;
    mul.wide.u32 %rd3, %r3, %r2; add.u64 %rd4, %rd0, %rd3; ld.global.f32 %f1, [%rd4];
    fma.rn.f32 %f0, %f1, %f1, %f0; add.u32 %r3, %r3, 1; bra SUM;
NORM: cvt.rn.f32.u32 %f2, %r1; div.rn.f32 %f3, %f0, %f2;
    mov.f32 %f4, 0F{eps_hex:08X}; add.f32 %f5, %f3, %f4; rsqrt.approx.f32 %f6, %f5;
    mov.u32 %r3, 0;
WR: setp.lt.u32 %p0, %r3, %r1; @!%p0 bra DONE;
    mul.wide.u32 %rd3, %r3, %r2; add.u64 %rd4, %rd0, %rd3; add.u64 %rd5, %rd2, %rd3;
    ld.global.f32 %f7, [%rd4]; ld.global.f32 %f8, [%rd5]; add.u64 %rd4, %rd1, %rd3;
    mul.f32 %f9, %f7, %f6; mul.f32 %f7, %f9, %f8; st.global.f32 [%rd4], %f7;
    add.u32 %r3, %r3, 1; bra WR;
DONE: ret;
}}
"""

print("\n--- V2: Single-threaded (1 thread, sequential, no shuffle) ---")
out_gpu.zero_()
result, err = load_and_run_ptx(ptx_v2, "rmsnorm_v2", x_gpu, gamma_gpu, out_gpu, HIDDEN, 32)
if err:
    print(f"ERROR: {err}")
    results["V2: Single-threaded"] = None
else:
    cos = cosine_sim(cpu_out, result)
    print(f"cosine={cos:.10f}")
    results["V2: Single-threaded"] = cos

# ===== Variant 3: Shared memory reduction (no warp shuffles) =====
ptx_v3 = f"""
.version 8.0
.target sm_90
.address_size 64
.visible .entry rmsnorm_v3(
    .param .u64 input_ptr, .param .u64 output_ptr, .param .u64 gamma_ptr
) {{
    .reg .u64 %rd<6>; .reg .u32 %r<8>; .reg .f32 %f<10>; .reg .pred %p<2>;
    .shared .f32 smem[32];
    mov.u32 %r0, %tid.x;
    ld.param.u64 %rd0, [input_ptr]; ld.param.u64 %rd1, [output_ptr]; ld.param.u64 %rd2, [gamma_ptr];
    mov.u32 %r1, {HIDDEN}; mov.u32 %r2, 4; mov.f32 %f0, 0F00000000; mov.u32 %r3, 0;
SUM: add.u32 %r4, %r3, %r0; setp.lt.u32 %p0, %r4, %r1; @!%p0 bra RED;
    mul.wide.u32 %rd3, %r4, %r2; add.u64 %rd4, %rd0, %rd3; ld.global.f32 %f1, [%rd4];
    fma.rn.f32 %f0, %f1, %f1, %f0; add.u32 %r3, %r3, 32; bra SUM;
RED: mul.u32 %r5, %r0, 4; st.shared.f32 [%r5], %f0; bar.sync 0;
    mov.u32 %r6, 16;
R1: setp.eq.u32 %p0, %r6, 0; @%p0 bra RD;
    setp.lt.u32 %p1, %r0, %r6; @!%p1 bra R2;
    add.u32 %r7, %r0, %r6; mul.u32 %r7, %r7, 4; ld.shared.f32 %f1, [%r7];
    mul.u32 %r7, %r0, 4; ld.shared.f32 %f2, [%r7]; add.f32 %f2, %f2, %f1; st.shared.f32 [%r7], %f2;
R2: bar.sync 0; shr.u32 %r6, %r6, 1; bra R1;
RD: ld.shared.f32 %f3, [0];
    cvt.rn.f32.u32 %f4, %r1; div.rn.f32 %f5, %f3, %f4;
    mov.f32 %f6, 0F{eps_hex:08X}; add.f32 %f7, %f5, %f6; rsqrt.approx.f32 %f8, %f7;
    mov.u32 %r3, 0;
WR: add.u32 %r4, %r3, %r0; setp.lt.u32 %p0, %r4, %r1; @!%p0 bra DN;
    mul.wide.u32 %rd3, %r4, %r2; add.u64 %rd4, %rd0, %rd3; add.u64 %rd5, %rd2, %rd3;
    ld.global.f32 %f1, [%rd4]; ld.global.f32 %f2, [%rd5]; add.u64 %rd4, %rd1, %rd3;
    mul.f32 %f9, %f1, %f8; mul.f32 %f1, %f9, %f2; st.global.f32 [%rd4], %f1;
    add.u32 %r3, %r3, 32; bra WR;
DN: ret;
}}
"""

print("\n--- V3: Shared memory reduction (no warp shuffles) ---")
out_gpu.zero_()
result, err = load_and_run_ptx(ptx_v3, "rmsnorm_v3", x_gpu, gamma_gpu, out_gpu, HIDDEN, 32)
if err:
    print(f"ERROR: {err}")
    results["V3: Shared memory (no shuffle)"] = None
else:
    cos = cosine_sim(cpu_out, result)
    print(f"cosine={cos:.10f}")
    results["V3: Shared memory (no shuffle)"] = cos

# ===== Variant 4: PTX version 7.0 + target sm_75 =====
ptx_v4 = ptx_v1.replace(".version 8.0", ".version 7.0").replace(".target sm_90", ".target sm_75").replace("rmsnorm_v1", "rmsnorm_v4")

print("\n--- V4: PTX 7.0 + sm_75 target (oldest viable ISA) ---")
out_gpu.zero_()
result, err = load_and_run_ptx(ptx_v4, "rmsnorm_v4", x_gpu, gamma_gpu, out_gpu, HIDDEN, 32)
if err:
    print(f"ERROR: {err}")
    results["V4: PTX 7.0 + sm_75"] = None
else:
    cos = cosine_sim(cpu_out, result)
    print(f"cosine={cos:.10f}")
    results["V4: PTX 7.0 + sm_75"] = cos

# ===== Variant 5: mul+add instead of FMA, sqrt+rcp instead of rsqrt =====
ptx_v5 = f"""
.version 8.0
.target sm_90
.address_size 64
.visible .entry rmsnorm_v5(
    .param .u64 input_ptr, .param .u64 output_ptr, .param .u64 gamma_ptr
) {{
    .reg .u64 %rd<6>; .reg .u32 %r<5>; .reg .f32 %f<12>; .reg .pred %p<2>;
    mov.u32 %r0, %tid.x;
    ld.param.u64 %rd0, [input_ptr]; ld.param.u64 %rd1, [output_ptr]; ld.param.u64 %rd2, [gamma_ptr];
    mov.u32 %r1, {HIDDEN}; mov.u32 %r2, 4; mov.f32 %f0, 0F00000000; mov.u32 %r3, 0;
SUM: add.u32 %r4, %r3, %r0; setp.lt.u32 %p0, %r4, %r1; @!%p0 bra RED;
    mul.wide.u32 %rd3, %r4, %r2; add.u64 %rd4, %rd0, %rd3; ld.global.f32 %f1, [%rd4];
    mul.rn.f32 %f2, %f1, %f1; add.rn.f32 %f0, %f0, %f2;
    add.u32 %r3, %r3, 32; bra SUM;
RED: shfl.sync.down.b32 %f3, %f0, 16, 31, 0xFFFFFFFF; add.f32 %f0, %f0, %f3;
    shfl.sync.down.b32 %f3, %f0, 8, 31, 0xFFFFFFFF; add.f32 %f0, %f0, %f3;
    shfl.sync.down.b32 %f3, %f0, 4, 31, 0xFFFFFFFF; add.f32 %f0, %f0, %f3;
    shfl.sync.down.b32 %f3, %f0, 2, 31, 0xFFFFFFFF; add.f32 %f0, %f0, %f3;
    shfl.sync.down.b32 %f3, %f0, 1, 31, 0xFFFFFFFF; add.f32 %f0, %f0, %f3;
    shfl.sync.idx.b32 %f4, %f0, 0, 31, 0xFFFFFFFF;
    cvt.rn.f32.u32 %f5, %r1; div.rn.f32 %f6, %f4, %f5;
    mov.f32 %f7, 0F{eps_hex:08X}; add.f32 %f8, %f6, %f7;
    sqrt.rn.f32 %f9, %f8; rcp.rn.f32 %f10, %f9;
    mov.u32 %r3, 0;
WR: add.u32 %r4, %r3, %r0; setp.lt.u32 %p0, %r4, %r1; @!%p0 bra DN;
    mul.wide.u32 %rd3, %r4, %r2; add.u64 %rd4, %rd0, %rd3; add.u64 %rd5, %rd2, %rd3;
    ld.global.f32 %f1, [%rd4]; ld.global.f32 %f2, [%rd5]; add.u64 %rd4, %rd1, %rd3;
    mul.rn.f32 %f11, %f1, %f10; mul.rn.f32 %f1, %f11, %f2; st.global.f32 [%rd4], %f1;
    add.u32 %r3, %r3, 32; bra WR;
DN: ret;
}}
"""

print("\n--- V5: mul+add (no FMA), sqrt+rcp (no rsqrt) ---")
out_gpu.zero_()
result, err = load_and_run_ptx(ptx_v5, "rmsnorm_v5", x_gpu, gamma_gpu, out_gpu, HIDDEN, 32)
if err:
    print(f"ERROR: {err}")
    results["V5: No FMA, no rsqrt"] = None
else:
    cos = cosine_sim(cpu_out, result)
    print(f"cosine={cos:.10f}")
    results["V5: No FMA, no rsqrt"] = cos

# ===== Summary =====
print("\n" + "=" * 70)
print("SUMMARY: 5 PTX implementations via cuModuleLoadData on sm_121")
print("=" * 70)
for name, cos in results.items():
    if cos is None:
        s = "ERROR"
    elif cos > 0.9999:
        s = "PASS — PTX CAN be fixed this way"
    elif cos > 0.98:
        s = "MARGINAL"
    else:
        s = "FAIL — same JIT bug"
    print(f"  {name:45s} {f'cos={cos:.10f}' if cos else 'N/A':>20s}  {s}")

passing = [n for n,c in results.items() if c is not None and c > 0.9999]
failing = [n for n,c in results.items() if c is not None and c < 0.98]

if passing:
    print(f"\nFALSIFIED: {len(passing)} PTX variant(s) produce correct results!")
    print("The PTX CAN be fixed by using these approaches:")
    for n in passing:
        print(f"  ✓ {n}")
if failing:
    print(f"\n{len(failing)} variant(s) fail with the same JIT bug:")
    for n in failing:
        print(f"  ✗ {n}")
if not passing:
    print("\nNOT FALSIFIED: All PTX variants fail through cuModuleLoadData JIT.")
