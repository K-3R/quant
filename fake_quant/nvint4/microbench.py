"""
Config 3 (custom_qmatmul) vs 4 (qmatmul): kernel-level timings.

Modes:
  python microbench.py              print per-kernel CUDA time
  python microbench.py --nsys       NVTX only (clean nsys trace)
  python microbench.py --ncu        fixed launch counts for ncu

nsys:
    nsys profile -t cuda,nvtx,cublas --cuda-memory-usage=true \\
        -o results/nsys_report python microbench.py --nsys

ncu:
    After warmup: one layer (default c_fc), prefill + decode, both configs, 1 iter.

    /usr/local/cuda-12.8/bin/ncu --profile-from-start off --set full -f \\
        -o ncu/test python microbench.py --ncu

    python microbench.py --ncu --layer c_attn




"""
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.profiler import ProfilerActivity, profile
from qmatmul import sr_kernel

torch.manual_seed(0)
dev = "cuda"

# Tensors are FP32; cuBLAS GEMM uses TF32 (same as gpt2_quant.py).
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

PREFILL = [
    ("c_attn",     4096,  768, 2304),
    ("c_proj",     4096,  768,  768),
    ("c_fc",       4096,  768, 3072),
    ("mlp.c_proj", 4096, 3072,  768),
]
# Same K, N as prefill; M=1 is decode with KV cache.
DECODE = [(name, 1, K, N) for name, _, K, N in PREFILL]

CONFIGS = [
    (4, "qmatmul",        sr_kernel.qmatmul),
    (3, "custom_qmatmul", sr_kernel.custom_qmatmul),
]


def cuda_us(evt):
    if hasattr(evt, "self_device_time_total"):
        return evt.self_device_time_total
    return evt.self_cuda_time_total


def classify(name):
    n = name.lower()
    if "tensorabsmax" in n:
        return "TensorAbsMax"
    if "fakequantcoldir" in n:
        return "FakeQuantColDir"
    if "fakequantrowdir" in n:
        return "FakeQuantRowDir"
    if "matmul" in n and "gemm" not in n:
        return "MatMul"
    if any(k in n for k in ("gemm", "xmma", "cutlass", "ampere", "wmma", "cublas")):
        return "GEMM"
    if any(k in n for k in ("fill", "memset", "zero", "setzero")):
        return "memset/zeros"
    if any(k in n for k in ("copy", "contiguous", "memcpy")):
        return "copy"
    return "other"


def print_kernel_table(prof, tag):
    rows = []
    for evt in prof.key_averages():
        us = cuda_us(evt)
        if us <= 0:
            continue
        rows.append((classify(evt.key), evt.key, us, evt.count))
    rows.sort(key=lambda r: -r[2])
    total = sum(r[2] for r in rows) or 1.0
    nlaunch = max((r[3] for r in rows), default=1)

    grouped = {}
    gemm_names = []
    for cat, raw, us, cnt in rows:
        grouped[cat] = grouped.get(cat, 0.0) + us
        if cat == "GEMM":
            gemm_names.append((raw, us))

    quant = grouped.get("TensorAbsMax", 0) + grouped.get("FakeQuantColDir", 0)
    gemm  = grouped.get("GEMM", 0) + grouped.get("MatMul", 0)
    core  = quant + gemm

    print(f"\n=== {tag} ===")
    print(f"  {'category':<18} {'us/call':>10} {'share':>8}")
    for cat in ("TensorAbsMax", "FakeQuantColDir", "FakeQuantRowDir",
                "MatMul", "GEMM", "memset/zeros", "copy", "other"):
        if cat not in grouped:
            continue
        per = grouped[cat] / nlaunch
        print(f"  {cat:<18} {per:10.1f} {grouped[cat]/total*100:7.1f}%")
    print(f"  {'total':<18} {total/nlaunch:10.1f}  100.0%")
    if core > 0:
        print(f"  quant share of (quant+GEMM): {quant/core*100:.1f}%")
    if gemm_names:
        raw, _ = max(gemm_names, key=lambda x: x[1])
        print(f"  cuBLAS kernel: {raw}")


def make_inputs(M, K, N):
    X = torch.randn(M, K, device=dev)
    W = torch.randn(K, N, device=dev) * 0.02
    Wq, _ = sr_kernel.quantize_weight(W)
    torch.cuda.synchronize()
    return X, Wq


def run_print(warmup=10, iters=10):
    """Per-kernel CUDA times via torch.profiler (no nsys needed)."""
    print(f"{torch.cuda.get_device_name(0)}  TF32={torch.backends.cuda.matmul.allow_tf32}")
    for phase, shapes in (("prefill", PREFILL), ("decode", DECODE)):
        for name, M, K, N in shapes:
            X, Wq = make_inputs(M, K, N)
            for cfg, tag, fn in CONFIGS:
                for _ in range(warmup):
                    fn(X, Wq)
                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
                    for _ in range(iters):
                        fn(X, Wq)
                    torch.cuda.synchronize()
                print_kernel_table(prof, f"#{cfg} {tag} | {phase} {name} | M={M} K={K} N={N}")


def _ncu_profiler_start():
    """ncu --profile-from-start off hooks cuProfilerStart (driver), not always PyTorch cudart."""
    import ctypes
    lib = ctypes.CDLL("libcuda.so.1")
    lib.cuInit.argtypes, lib.cuInit.restype = [ctypes.c_uint], ctypes.c_int
    lib.cuProfilerStart.argtypes, lib.cuProfilerStart.restype = [], ctypes.c_int
    e0 = lib.cuInit(0)
    e1 = lib.cuProfilerStart()
    print(f"cuProfilerStart: cuInit={e0} start={e1} (0=ok)", flush=True)
    if e1 != 0:
        raise RuntimeError(f"cuProfilerStart failed: {e1}")


def _ncu_profiler_stop():
    import ctypes
    lib = ctypes.CDLL("libcuda.so.1")
    lib.cuProfilerStop.argtypes, lib.cuProfilerStop.restype = [], ctypes.c_int
    e = lib.cuProfilerStop()
    print(f"cuProfilerStop: {e} (0=ok)", flush=True)


def _shapes_named(layer):
    pre = [s for s in PREFILL if s[0] == layer]
    dec = [s for s in DECODE if s[0] == layer]
    if not pre:
        raise ValueError(f"unknown layer {layer}; choose {[s[0] for s in PREFILL]}")
    return pre, dec


def run_nvtx(warmup=10, iters=20, profile_measured=False, layer=None):
    """Prefill/decode + config 3/4 with scoped NVTX (nsys and ncu)."""
    phases = (("prefill", PREFILL), ("decode", DECODE))
    if layer is not None:
        pre, dec = _shapes_named(layer)
        phases = (("prefill", pre), ("decode", dec))

    print("=== warmup ===", flush=True)
    with torch.cuda.nvtx.range("warmup"):
        Xw, Wqw = make_inputs(4096, 768, 768)
        for _ in range(warmup):
            sr_kernel.qmatmul(Xw, Wqw, "warmup:#4")
            sr_kernel.custom_qmatmul(Xw, Wqw, "warmup:#3")
        torch.cuda.synchronize()

    print("=== measured ===", flush=True)
    if profile_measured:
        _ncu_profiler_start()
    for phase, shapes in phases:
        with torch.cuda.nvtx.range(phase):
            for name, M, K, N in shapes:
                X, Wq = make_inputs(M, K, N)
                for cfg, tag, fn in CONFIGS:
                    scope = f"#{cfg}:{phase}:{name}"
                    print(f"  {scope} M={M}", flush=True)
                    with torch.cuda.nvtx.range(scope):
                        for _ in range(iters):
                            fn(X, Wq, scope)
                    torch.cuda.synchronize()
    if profile_measured:
        torch.cuda.synchronize()
        _ncu_profiler_stop()
    print("=== done ===", flush=True)


def run_nsys():
    run_nvtx()


def run_ncu(layer="c_fc"):
    run_nvtx(warmup=10, iters=1, profile_measured=True, layer=layer)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nsys", action="store_true", help="NVTX-only run for nsys")
    p.add_argument("--ncu", action="store_true", help="short capture for ncu")
    p.add_argument("--layer", default="c_fc",
                   help="ncu: one GPT-2 layer name (prefill+decode). default c_fc")
    args = p.parse_args()
    if args.ncu:
        run_ncu(layer=args.layer)
    elif args.nsys or os.environ.get("NSYS_PROFILING_SESSION_ID"):
        run_nsys()
    else:
        run_print()


if __name__ == "__main__":
    main()
