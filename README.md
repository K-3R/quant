# CUDA Group-wise INT4 Fake-Quantization Kernel

A PyTorch extension implementing group-wise INT4 fake-quantization in CUDA.
Activations and weights are quantized along K (input-channel direction) in groups of 16,
using a two-level scale: a per-group scale stored in FP8 (E4M3) multiplied by a per-tensor scale.

## Goals

1. **Minimal quantization overhead** relative to an FP32 `torch.matmul` baseline.
2. **Faster than PyTorch-native fake-quantization.** The PyTorch path (`amax` → `div` → `round` → `clamp` → `mul`) launches a separate kernel per op and round-trips the tensor through HBM each time. Fusing this into a single kernel should win on latency.

## Configurations Compared

| # | Quantization | GEMM |
|---|---|---|
| 1 | none (FP32 baseline) | cuBLAS SGEMM |
| 2 | PyTorch elementwise chain | cuBLAS SGEMM |
| 3 | CUDA fused kernel | custom tiled shared-memory GEMM |
| 4 | CUDA fused kernel | cuBLAS SGEMM |

Config 3 vs 4 isolates the GEMM contribution; config 2 vs 4 isolates the quantization contribution.
The custom `MatMul` kernel (16×16 tiles, shared memory) is kept in the tree for this comparison —
cuBLAS won on latency, so config 4 is the shipped path.

## Pipeline

```
quantize_weight(W)          # once, at model load
 ├─ TensorAbsMax            # per-tensor absmax → s_t
 └─ FakeQuantRowDir         # per-group absmax → FP8 s_g → quant/dequant

qmatmul(X, Wq)              # every forward
 ├─ TensorAbsMax
 ├─ FakeQuantColDir
 └─ at::matmul              # cuBLAS SGEMM
```

Group size 16, symmetric INT4 `[-7, +7]`, `s = s_g * s_t` where `s_t = tensor_absmax / E4M3_MAX / QMAX`.

## Methodology

### Nsight Systems — end-to-end latency

```bash
nsys profile -t cuda,nvtx,cublas --cuda-memory-usage=true -o report python bench.py
```

Tracked: fraction of `qmatmul` spent in quantization kernels (the headline number), which cuBLAS
kernel `at::matmul` selects (SGEMM vs TF32 path), inter-kernel gaps from per-forward allocation and
the `zeros()` memset, and launch overhead as kernels shrink. Measured separately for prefill-shaped
(large M) and decode-shaped (small M) inputs, since the quantization share grows as M drops.

### Nsight Compute — per-kernel metrics

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

## Results

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

## Known Issues / In Progress

**`TensorAbsMax` — shared-atomic serialization.** All 256 threads per block `atomicMax` into a single
shared variable, serializing up to 256 ways. Moving to thread-local max → warp shuffle → one shared
atomic per warp → one global atomic per block.

**`FakeQuantRowDir` — poor coalescing.** With block shape `(AX=4, AY=64)`, a warp spans
`threadIdx.x = 0..3` and `threadIdx.y = 0..7`, so each warp touches 8 different rows at 16B each and
over-fetches sectors. Reworking so x covers contiguous addresses, with the group reduction as a y-loop.

**Warp divergence in group reduction.** Both fake-quant kernels run a 16-iteration loop under
`if (lane == 0)`, idling 15/16 threads. Replacing with a 4-stage `__shfl_xor_sync` reduction, which
also removes one `__syncthreads()`.

**Redundant passes and allocation.** `TensorAbsMax` and `FakeQuant*` each read the input tensor, and
the per-forward `tmax` tensor triggers a separate memset kernel. Investigating single-pass fusion and
buffer reuse.

## Next Steps

This is *fake* quantization: error is simulated and values are restored to FP32 before an FP32 GEMM,
so neither compute nor memory footprint actually drops. The current objective is accuracy simulation
at minimal overhead.

Real INT4 low-precision compute is next:

- INT4-packed weight storage (2 elements/byte) for genuine memory-traffic and footprint reduction
- Dequantization fused into the GEMM prologue (mixed-input INT4 × FP16 GEMM)
- Tensor Core (MMA) path — at which point the figure of merit shifts from bandwidth to
  `sm__pipe_tensor_op_*_cycles_active`
- Benchmarking against CUTLASS mixed-input GEMM, Marlin, and Machete

## Usage

```python
import torch
from torch.utils.cpp_extension import load

ext = load(name="qkernel", sources=["binding.cpp", "kernel.cu"], verbose=True)

Wq, w_scale = ext.quantize_weight(W)   # once, at load
Y = ext.qmatmul(X, Wq)                 # per forward
```

Requires `K % GROUP_SIZE == 0` and FP32 CUDA tensors.
