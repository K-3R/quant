# CUDA Group-wise INT4 Fake-Quantization Kernel

A PyTorch extension implementing group-wise INT4 fake-quantization in CUDA.
Activations and weights are quantized along K (input-channel direction) in groups of 16,
using a two-level scale: a per-group scale stored in FP8 (E4M3) multiplied by a per-tensor scale.

## 💡 Goals

1. **Minimal quantization overhead** relative to an cuBLAS TF32 `torch.matmul` baseline.
2. **Faster than PyTorch-native fake-quantization.** The PyTorch path (`amax` → `div` → `round` → `clamp` → `mul`) launches a separate kernel per op and round-trips the tensor through HBM each time. Fusing this into a single kernel should win on latency.

## 1️⃣ Configurations Compared

| # | Label | Quantization | GEMM |
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

| | | | tf32 | ref-fq | tiled | kernel | 4/1 | 4/2 | 4/3 |
|---|---|---|---|---|---|---|---|---|---|
| B=1 | S=512 | M=512 | 15.60 | 122.53 | 78.71 | **17.92** | ×0.87 | ×6.84 | ×4.39 |
| B=8 | S=512 | M=4096 | 103.03 | 226.83 | 551.76 | **126.25** | ×0.82 | ×1.80 | ×4.37 |
| B=16 | S=1024 | M=16384 | 430.80 | 669.64 | 2335.29 | **516.30** | ×0.83 | ×1.30 | ×4.52 |

### 📍 Decode (ms/token, KV cache, M=1)

| | | tf32 | ref-fq | tiled | kernel | 4/1 | 4/2 | 4/3 |
|---|---|---|---|---|---|---|---|---|
| B=1 | prompt=128 | 3.80 | 63.94 | 8.79 | **4.36** | ×0.87 | ×14.67 | ×2.02 |
| B=8 | prompt=128 | 5.39 | 94.26 | 10.21 | **6.47** | ×0.83 | ×14.56 | ×1.58 |

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
nsys profile -t cuda,nvtx,cublas --cuda-memory-usage=true -o report python microbench.py
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
ncu --set full --target-processes all -k "regex:TensorAbsMax|FakeQuant" \
    --import-source yes -o prof python bench.py
```

All three quantization kernels are memory-bound (4B read / 4B write per element), so effective
bandwidth — not FLOPS — is the figure of merit.

| Priority | Metric | Target |
|---|---|---|
| 1 | `dram__throughput.avg.pct_of_peak_sustained_elapsed` | ≥ 70–80% of peak |
| 1 | `dram__bytes_{read,write}.sum` | compare against theoretical minimum traffic |
| 2 | `l1tex__t_sectors_…op_ld.sum / l1tex__t_requests_…op_ld.sum` | 4 sectors/request (coalesced) |
| 3 | `smsp__average_warps_issue_stalled_*_per_issue_active.ratio` | isolate `long_scoreboard` / `mio_throttle` / `barrier` |
| 3 | `l1tex__t_set_accesses_pipe_lsu_mem_shared_op_atom.sum` | shared-atomic contention |
| 4 | `smsp__thread_inst_executed_per_inst_executed.ratio` | → 32 (no divergence) |
| 4 | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_*` | 0 |
| 5 | `sm__warps_active.avg.pct_of_peak_sustained_active` | achieved vs theoretical occupancy |

Occupancy is checked last: if bandwidth is low while occupancy is adequate, the cause is the access
pattern or serialization, not occupancy.

Measurement notes: ≥10 warmup iterations before timing; `torch.backends.cuda.matmul.allow_tf32`
pinned to the same value across all configs; `--replay-mode application` used to cross-check
anomalous ncu results.

## 5️⃣ Results

> Environment: _GPU / CUDA / PyTorch versions_ — Shapes: _M, K, N_

| Config | Latency (µs) | vs. FP32 baseline |
|---|---|---|
| 1. FP32 baseline | | 1.00× |
| 2. PyTorch fake-quant + cuBLAS | | |
| 3. Fused kernel + tiled GEMM | | |
| 4. Fused kernel + cuBLAS | | |

| Kernel | Latency (µs) | Share of `qmatmul` |
|---|---|---|
| `TensorAbsMax` | | |
| `FakeQuantColDir` | | |
| GEMM | | |

| Kernel | DRAM BW (% peak) | Sectors/req | Achieved occupancy | Dominant stall |
|---|---|---|---|---|
| `TensorAbsMax` | | | | |
| `FakeQuantColDir` | | | | |
| `FakeQuantRowDir` | | | | |

## 6️⃣ Next Steps

This is *fake* quantization: error is simulated and values are restored to FP32 before an FP32 GEMM,
so neither compute nor memory footprint actually drops. The current objective is accuracy simulation
at minimal overhead.

Real INT4 low-precision compute is next:

- INT4-packed weight storage (2 elements/byte) for genuine memory-traffic and footprint reduction
- Dequantization fused into the GEMM prologue (mixed-input INT4 × FP16 GEMM)
- Tensor Core (MMA) path — at which point the figure of merit shifts from bandwidth to
  `sm__pipe_tensor_op_*_cycles_active`
- Benchmarking against CUTLASS mixed-input GEMM, Marlin, and Machete

## 7️⃣ Usage

```python
import torch
from torch.utils.cpp_extension import load

ext = load(name="qkernel", sources=["binding.cpp", "kernel.cu"], verbose=True)

Wq, w_scale = ext.quantize_weight(W)   # once, at load
Y = ext.qmatmul(X, Wq)                 # per forward
```

Requires `K % GROUP_SIZE == 0` and FP32 CUDA tensors.
