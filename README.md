# CUDA Group-wise INT4 Fake-Quantization Kernel

A PyTorch extension implementing group-wise INT4 fake-quantization in CUDA.
Activations and weights are quantized along K (input-channel direction) in groups of 16,
using a two-level scale: a per-group scale stored in FP8 (E4M3) multiplied by a per-tensor scale.

## 💡 Goals

1. **Minimal quantization overhead** relative to an cuBLAS TF32 `torch.matmul` baseline.
2. **Faster than PyTorch-native fake-quantization.** The PyTorch path (`amax` → `div` → `round` → `clamp` → `mul`) launches a separate kernel per op and round-trips the tensor through HBM each time. Fusing this into a single kernel should win on latency.

## 1️⃣ Configurations Compared

| Config | Label | Quantization | GEMM |
|---|---|---|---|
| 1 | `tf32` | none (baseline) | cuBLAS SGEMM |
| 2 | `ref-fq` | PyTorch elementwise chain | cuBLAS SGEMM |
| 3 | `tiled` | CUDA fused kernel | custom tiled shared-memory GEMM (`custom_qmatmul`) |
| 4 | `kernel` | CUDA fused kernel | cuBLAS SGEMM (`qmatmul`) |

## 2️⃣ Code Hierarchy

```
quantize_weight(W)            # once, at model load
 ├─ TensorAbsMax              # per-tensor absmax → FP32 tensor scale
 └─ FakeQuantRowDir           # per-group absmax → FP8 group scale → quant/dequant
                              # returns (Wq, scale)

custom_qmatmul(X, Wq)         # config 3
 ├─ TensorAbsMax              # per-tensor absmax over X
 ├─ FakeQuantColDir           # per-group absmax → FP8 group scale → quant/dequant
 └─ MatMul                    # custom 16×16 tiled shared-memory GEMM

qmatmul(X, Wq)                # config 4
 ├─ TensorAbsMax
 ├─ FakeQuantColDir
 └─ at::matmul                # cuBLAS SGEMM

```

## 3️⃣ Results

Measured on GPT-2, TF32, batched end-to-end. Ratio columns are the speedup of config 4 over the
indicated config — values below 1.00 mean config 4 is slower.

> Environment: NVIDIA RTX A6000 · CUDA 12.1 · PyTorch 2.5.1+cu121

### 📍 Prefill (ms)

| | | | tf32 | ref-fq | Config 3 | Config 4 |
|---|---|---|---|---|---|---|
| B=1 | S=512 | M=512 | 10.51 | 28.47 | 20.38 | **12.02** |
| B=8 | S=512 | M=4096 | 63.39 | 107.22 | 125.75 | **77.84** |
| B=16 | S=1024 | M=16384 | 256.95 | 409.15 | 499.08 | **311.07** |

### 📍 Decode (ms/token, KV cache, M=1)

| | | tf32 | ref-fq | Config 3 | Config 4 |
|---|---|---|---|---|---|
| B=1 | prompt=128 | 3.38 | 18.98 | 8.38 | **3.80** |
| B=8 | prompt=128 | 3.43 | 20.75 | 9.42 | **4.58** |

### 💬 Reading the numbers

**Goal 1 — Quantization overhead.** Quantization costs 15–22% over the TF32 baseline (×0.87 to ×0.82).

**Goal 2 — vs. PyTorch fake-quant.** Config 4 wins in every case, but the margin is strongly
shape-dependent. At prefill the advantage shrinks as M grows (×6.84 → ×1.80 → ×1.30): the GEMM
dominates at large M, so the per-op HBM round-trips of the PyTorch chain amortize away. At decode the
advantage jumps to ~×14.6, because with M=1 there is almost no GEMM to hide behind and the PyTorch
path is paying almost pure kernel-launch and memory-round-trip overhead.

**Why cuBLAS is the shipped GEMM.** The custom tiled kernel loses by ×4.4–4.5 at prefill — a plain
16×16 shared-memory tiling has no register blocking, no vectorized loads, and no Tensor Core path, so
it falls far short of cuBLAS on compute-bound shapes. At decode the gap compresses to ×1.6–2.0:
with M=1 the operation is effectively a GEMV and memory-bound regardless of tiling, so there is much
less for a tuned kernel to win. The tiled kernel is kept in the tree as config 3 rather than removed,
since it is what makes the GEMM contribution separable from the quantization contribution.


## 4️⃣ Methodology

### 📍 Naive Evaluation

```bash
python microbench.py
```

Latency is measured with `torch.profiler`, which uses CUPTI to capture the name and duration of each GPU kernel as it completes.

```text
=== #4 qmatmul | prefill c_attn | M=4096 K=768 N=2304 ===
  category            us/call    share
  TensorAbsMax           26.7     4.5%
  FakeQuantColDir        51.7     8.7%
  GEMM                  512.9    86.5%
  memset/zeros            1.5     0.3%
  total                 592.8   100.0%
  quant share of (quant+GEMM): 13.3%
  cuBLAS kernel: cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4

=== #3 custom_qmatmul | prefill c_attn | M=4096 K=768 N=2304 ===
  category            us/call    share
  TensorAbsMax           24.3     0.3%
  FakeQuantColDir        54.3     0.7%
  MatMul               8074.3    99.0%
  memset/zeros            1.6     0.0%
  total                8154.5   100.0%
  quant share of (quant+GEMM): 1.0%
```


### 📍 Nsight Systems — where the time goes

```bash
nsys profile -t cuda,nvtx,cublas --cuda-memory-usage=true --force-overwrite=true -o nsys/report python microbench.py
```

Tracked: fraction of `qmatmul` spent in quantization kernels (the headline number), which cuBLAS
kernel `at::matmul` selects (SGEMM vs TF32 path), inter-kernel gaps from per-forward allocation and
the `zeros()` memset, and launch overhead as kernels shrink. Measured separately for prefill-shaped
(large M) and decode-shaped (small M) inputs, since the quantization share grows as M drops.


```text
Python NVTX:  "#4 qmatmul"
  └─ C++ NvtxRange: "qmatmul"
       ├─ "alloc+zeros"          # empty(Xq, scale) + zeros(tmax)
       ├─ "TensorAbsMax"
       ├─ "FakeQuantColDir"
       └─ "cuBLAS GEMM"          # at::matmul → TF32 tensor-core

Python NVTX:  "#3 custom_qmatmul"
  └─ C++ NvtxRange: "custom_qmatmul"
       ├─ "alloc+zeros"          # empty(Xq, scale, C) + zeros(tmax)
       ├─ "TensorAbsMax"
       ├─ "FakeQuantColDir"
       └─ "MatMul"               # launch_matmul, 16×16 tiled FP32
```
### 📍 Nsight Compute — per-kernel metrics

```bash
ncu --profile-from-start off --set full -f -o ncu/test python microbench.py --ncu --layer c_fc
```

### MatMul
| Metric | Value |
|---|---|
| Duration [us] | 68.64 |
| SM Throughput [%] | 22.13 |
| Memory Throughput [%] | 26.79 |
| L1 Throughput [%] | 49.11 |
| L2 Throughput [%] | 6.96 |
| DRAM Throughput [%] | 21.97 |

SM, DRAM and L1 throughput are below 50%. Therefore, the MatMul kernel is latency bound.

### TensorAbsMax (Tensor Scaling Factor)
| Metric | Value |
|---|---|
| Duration [us] | 21.82 |
| SM Throughput [%] | 36.43 |
| Memory Throughput [%] | 70.10 |
| L1 Throughput [%] | 38.01 |
| L2 Throughput [%] | 22.05 |
| DRAM Throughput [%] | 70.10 |

TensorAbsMax is DRAM bound.

### FakeQuantRowDir (Group Quant Scaling Factor)
| Metric | Value |
|---|---|
| Duration [us] | 45.70 |
| SM Throughput [%] | 63.20 |
| L1 Throughput [%] | 69.00 |
| L2 Throughput [%] | 62.09 |
| DRAM Throughput [%] | 54.86 |

FakeQuantRowDir is L1 bound.
