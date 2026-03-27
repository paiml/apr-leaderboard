#!/usr/bin/env python3
"""GH-559: Isolate whether the bug is in buffer allocation or kernel launch.

Test our exact PTX RMSNorm kernel with cuMemAlloc buffers (like trueno-gpu)
instead of PyTorch buffers. If it still works, the bug is in how trueno-gpu
sets up the CUDA context or passes kernel args.

Usage: uv run --with torch scripts/isolate-cuda-bug.py
"""
import torch
import ctypes
import math
import struct
import sys

HIDDEN = 3584
EPS = 1e-6
eps_hex = struct.unpack('I', struct.pack('f', EPS))[0]

libcuda = ctypes.CDLL("libcuda.so.1")

# Init CUDA via PyTorch (creates context)
torch.cuda.init()
_ = torch.zeros(1, device="cuda")

torch.manual_seed(42)
x_data = (torch.randn(HIDDEN) * 0.02).tolist()
gamma_data = (torch.randn(HIDDEN) * 0.5 + 1.0).tolist()

# CPU reference
sq = sum(v*v for v in x_data)
rms = math.sqrt(sq / HIDDEN + EPS)
cpu_out = [(x_data[i] / rms) * gamma_data[i] for i in range(HIDDEN)]

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CPU rms={rms:.10f}, sq_sum={sq:.10f}")

# Our exact PTX (same as trueno-gpu generates)
ptx = f"""
.version 8.0
.target sm_90
.address_size 64
.visible .entry rmsnorm(
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

def cosine_sim(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    return dot/(na*nb) if na*nb > 1e-12 else 0

# Load module
module = ctypes.c_void_p()
ptx_bytes = ptx.encode() + b'\0'
err = libcuda.cuModuleLoadData(ctypes.byref(module), ptx_bytes)
assert err == 0, f"cuModuleLoadData failed: {err}"
func = ctypes.c_void_p()
err = libcuda.cuModuleGetFunction(ctypes.byref(func), module, b"rmsnorm")
assert err == 0, f"cuModuleGetFunction failed: {err}"

def pack_f32_array(data):
    return struct.pack(f'{len(data)}f', *data)

def unpack_f32_array(buf, n):
    return list(struct.unpack(f'{n}f', buf))

# ===== Test A: PyTorch-allocated buffers =====
print("\n--- Test A: PyTorch buffers (torch.tensor.cuda) ---")
x_gpu_a = torch.tensor(x_data, dtype=torch.float32, device="cuda")
g_gpu_a = torch.tensor(gamma_data, dtype=torch.float32, device="cuda")
o_gpu_a = torch.zeros(HIDDEN, dtype=torch.float32, device="cuda")

ptr_x = ctypes.c_uint64(x_gpu_a.data_ptr())
ptr_o = ctypes.c_uint64(o_gpu_a.data_ptr())
ptr_g = ctypes.c_uint64(g_gpu_a.data_ptr())
args_a = (ctypes.c_void_p * 3)(
    ctypes.cast(ctypes.pointer(ptr_x), ctypes.c_void_p),
    ctypes.cast(ctypes.pointer(ptr_o), ctypes.c_void_p),
    ctypes.cast(ctypes.pointer(ptr_g), ctypes.c_void_p),
)
err = libcuda.cuLaunchKernel(func, 1,1,1, 32,1,1, 0, None, args_a, None)
assert err == 0, f"launch failed: {err}"
libcuda.cuCtxSynchronize()
out_a = o_gpu_a.cpu().tolist()
cos_a = cosine_sim(cpu_out, out_a)
print(f"cosine={cos_a:.10f}")

# ===== Test B: cuMemAlloc buffers (like trueno-gpu) =====
print("\n--- Test B: cuMemAlloc buffers (like trueno-gpu) ---")
nbytes = HIDDEN * 4

x_dev = ctypes.c_uint64()
g_dev = ctypes.c_uint64()
o_dev = ctypes.c_uint64()
assert libcuda.cuMemAlloc_v2(ctypes.byref(x_dev), nbytes) == 0
assert libcuda.cuMemAlloc_v2(ctypes.byref(g_dev), nbytes) == 0
assert libcuda.cuMemAlloc_v2(ctypes.byref(o_dev), nbytes) == 0

# Upload data
x_buf = pack_f32_array(x_data)
g_buf = pack_f32_array(gamma_data)
assert libcuda.cuMemcpyHtoD_v2(x_dev, x_buf, nbytes) == 0
assert libcuda.cuMemcpyHtoD_v2(g_dev, g_buf, nbytes) == 0

# Zero output
o_zeros = b'\x00' * nbytes
assert libcuda.cuMemcpyHtoD_v2(o_dev, o_zeros, nbytes) == 0

# Launch with cuMemAlloc pointers
ptr_xb = ctypes.c_uint64(x_dev.value)
ptr_ob = ctypes.c_uint64(o_dev.value)
ptr_gb = ctypes.c_uint64(g_dev.value)
args_b = (ctypes.c_void_p * 3)(
    ctypes.cast(ctypes.pointer(ptr_xb), ctypes.c_void_p),
    ctypes.cast(ctypes.pointer(ptr_ob), ctypes.c_void_p),
    ctypes.cast(ctypes.pointer(ptr_gb), ctypes.c_void_p),
)
err = libcuda.cuLaunchKernel(func, 1,1,1, 32,1,1, 0, None, args_b, None)
assert err == 0, f"launch failed: {err}"
libcuda.cuCtxSynchronize()

# Download result
out_buf = ctypes.create_string_buffer(nbytes)
assert libcuda.cuMemcpyDtoH_v2(out_buf, o_dev, nbytes) == 0
out_b = unpack_f32_array(out_buf.raw, HIDDEN)
cos_b = cosine_sim(cpu_out, out_b)
print(f"cosine={cos_b:.10f}")

# ===== Test C: PADDED buffer (trueno-gpu pads to 256 alignment) =====
print("\n--- Test C: Padded buffer (pad to 3840 = 15*256, like trueno pad_and_upload) ---")
PADDED = ((HIDDEN + 255) // 256) * 256  # 3584 → 3584 (already aligned)
# But what if trueno pads to NEXT multiple? Let's test 3840
PADDED2 = 3840  # 15 * 256
nbytes_pad = PADDED2 * 4

x_padded = x_data + [0.0] * (PADDED2 - HIDDEN)
x_pad_buf = pack_f32_array(x_padded)

x_dev_c = ctypes.c_uint64()
o_dev_c = ctypes.c_uint64()
assert libcuda.cuMemAlloc_v2(ctypes.byref(x_dev_c), nbytes_pad) == 0
assert libcuda.cuMemAlloc_v2(ctypes.byref(o_dev_c), nbytes_pad) == 0
assert libcuda.cuMemcpyHtoD_v2(x_dev_c, x_pad_buf, nbytes_pad) == 0
assert libcuda.cuMemcpyHtoD_v2(o_dev_c, b'\x00' * nbytes_pad, nbytes_pad) == 0

# NOTE: kernel uses HIDDEN=3584 as loop bound, NOT 3840
# So padding shouldn't matter. But let's verify.
ptr_xc = ctypes.c_uint64(x_dev_c.value)
ptr_oc = ctypes.c_uint64(o_dev_c.value)
args_c = (ctypes.c_void_p * 3)(
    ctypes.cast(ctypes.pointer(ptr_xc), ctypes.c_void_p),
    ctypes.cast(ctypes.pointer(ptr_oc), ctypes.c_void_p),
    ctypes.cast(ctypes.pointer(ptr_gb), ctypes.c_void_p),  # reuse gamma
)
err = libcuda.cuLaunchKernel(func, 1,1,1, 32,1,1, 0, None, args_c, None)
assert err == 0
libcuda.cuCtxSynchronize()
out_buf_c = ctypes.create_string_buffer(nbytes)
assert libcuda.cuMemcpyDtoH_v2(out_buf_c, o_dev_c, nbytes) == 0
out_c = unpack_f32_array(out_buf_c.raw, HIDDEN)
cos_c = cosine_sim(cpu_out, out_c)
print(f"cosine={cos_c:.10f}  (padded to {PADDED2})")

# ===== Test D: Different CUDA stream (non-default) =====
print("\n--- Test D: Non-default CUDA stream ---")
stream = ctypes.c_void_p()
assert libcuda.cuStreamCreate(ctypes.byref(stream), 0) == 0

o_dev_d = ctypes.c_uint64()
assert libcuda.cuMemAlloc_v2(ctypes.byref(o_dev_d), nbytes) == 0
assert libcuda.cuMemcpyHtoD_v2(o_dev_d, o_zeros, nbytes) == 0

ptr_od = ctypes.c_uint64(o_dev_d.value)
args_d = (ctypes.c_void_p * 3)(
    ctypes.cast(ctypes.pointer(ptr_xb), ctypes.c_void_p),
    ctypes.cast(ctypes.pointer(ptr_od), ctypes.c_void_p),
    ctypes.cast(ctypes.pointer(ptr_gb), ctypes.c_void_p),
)
err = libcuda.cuLaunchKernel(func, 1,1,1, 32,1,1, 0, stream, args_d, None)
assert err == 0
libcuda.cuStreamSynchronize(stream)
out_buf_d = ctypes.create_string_buffer(nbytes)
assert libcuda.cuMemcpyDtoH_v2(out_buf_d, o_dev_d, nbytes) == 0
out_d = unpack_f32_array(out_buf_d.raw, HIDDEN)
cos_d = cosine_sim(cpu_out, out_d)
print(f"cosine={cos_d:.10f}")

# ===== Test E: Multiple kernel launches (context reuse) =====
print("\n--- Test E: 10 sequential launches (context reuse stress) ---")
cos_worst = 1.0
for i in range(10):
    assert libcuda.cuMemcpyHtoD_v2(o_dev, o_zeros, nbytes) == 0
    err = libcuda.cuLaunchKernel(func, 1,1,1, 32,1,1, 0, None, args_b, None)
    assert err == 0
    libcuda.cuCtxSynchronize()
    assert libcuda.cuMemcpyDtoH_v2(out_buf, o_dev, nbytes) == 0
    out_e = unpack_f32_array(out_buf.raw, HIDDEN)
    cos_e = cosine_sim(cpu_out, out_e)
    if cos_e < cos_worst:
        cos_worst = cos_e
print(f"cosine={cos_worst:.10f} (worst of 10 runs)")

# ===== Summary =====
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
tests = {
    "A: PyTorch buffers": cos_a,
    "B: cuMemAlloc buffers": cos_b,
    "C: Padded buffer (3840)": cos_c,
    "D: Non-default stream": cos_d,
    "E: 10x sequential (stress)": cos_worst,
}
for name, cos in tests.items():
    s = "PASS" if cos > 0.9999 else "FAIL"
    print(f"  {name:40s} cosine={cos:.10f}  {s}")

if all(c > 0.9999 for c in tests.values()):
    print("\nALL PASS — cuModuleLoadData + cuLaunchKernel work correctly")
    print("with both PyTorch and cuMemAlloc buffers, padded buffers,")
    print("non-default streams, and repeated launches.")
    print()
    print("The bug must be in trueno-gpu's HIGHER-LEVEL code:")
    print("  - Weight upload (preload_weights_gpu)")
    print("  - Buffer management (GpuBuffer lifecycle)")
    print("  - Kernel parameter marshaling in CudaStream::launch_kernel")
    print("  - Multi-kernel pipeline (earlier kernel corrupts buffer)")
