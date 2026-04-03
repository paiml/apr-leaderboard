#!/usr/bin/env python3
"""Remap LoRA adapter tensor names from wgpu training format to GGUF merge format.

wgpu training saves:  layer.N.proj.lora_a
GGUF merge expects:   model.layers.N.self_attn.proj.lora_a  (for qkvo projections)
                      model.layers.N.mlp.proj.lora_a        (for gate/up/down projections)

Usage: python3 scripts/remap-adapter-tensors.py adapter.safetensors remapped.safetensors

This script fixes a naming mismatch between entrenar's wgpu training output
and the GGUF-based merge path. Upstream fix tracked in aprender.
"""
import sys
import struct
import json

ATTN_PROJS = {"q_proj", "k_proj", "v_proj", "o_proj"}
MLP_PROJS = {"gate_proj", "up_proj", "down_proj"}


def remap_name(name):
    # layer.N.proj.lora_{a|b} -> model.layers.N.{self_attn|mlp}.proj.weight.lora_{a|b}
    # The merge code constructs "{base_tensor_name}.lora_a" where
    # base_tensor_name = "model.layers.N.self_attn.q_proj.weight"
    parts = name.split(".")
    if len(parts) != 4 or parts[0] != "layer":
        return name
    layer_idx, proj, lora_part = parts[1], parts[2], parts[3]
    if proj in ATTN_PROJS:
        return f"model.layers.{layer_idx}.self_attn.{proj}.weight.{lora_part}"
    elif proj in MLP_PROJS:
        return f"model.layers.{layer_idx}.mlp.{proj}.weight.{lora_part}"
    return name


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.safetensors> <output.safetensors>")
        sys.exit(1)

    src, dst = sys.argv[1], sys.argv[2]

    with open(src, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
        data = f.read()

    new_header = {}
    remapped = 0
    for key, info in header.items():
        if key == "__metadata__":
            new_header[key] = info
            continue
        new_key = remap_name(key)
        if new_key != key:
            remapped += 1
        new_header[new_key] = info

    header_bytes = json.dumps(new_header).encode("utf-8")
    with open(dst, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(data)

    total = len(header) - (1 if "__metadata__" in header else 0)
    print(f"Remapped {remapped}/{total} tensors -> {dst}")


if __name__ == "__main__":
    main()
