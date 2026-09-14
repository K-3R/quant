"""
qkernel_v3 마이크로벤치 (ncu 용).

구조:
    ┌─ Global warmup ─┐   ← 캡처 안 함 (driver / cache 워밍업)
    │  1x quantize_weight  = 2 matching kernels  (TensorAbsMax, FakeQuantRowDir)
    │  3x qmatmul          = 9 matching kernels  (TensorAbsMax, FakeQuantColDir, sgemm) × 3
    └──────────────────┘   =  11 matching kernels 총합

    ┌─ Measured ──────┐   ← 여기부터 캡처
    │  각 shape 별 quantize_weight(2) + qmatmul(3) = 5
    │  4 shapes × 5 = 20 matching kernels
    └──────────────────┘

ncu 커맨드:
    ncu --target-processes all --set full \
        -k "FakeQuantColDir|FakeQuantRowDir|TensorAbsMax|[gG]emm|xmma" \
        --launch-skip-before-match 11 \
        -c 20 \
        -o results/fq_report python microbench.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from qmatmul import sr_kernel

torch.manual_seed(0)
dev = "cuda"

# gpt2_quant.py 와 동일하게 TF32 텐서코어 사용
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def one_pass(name, M, K, N):
    """quantize_weight(2 kernels) + qmatmul(3 kernels) = 5 matching launches."""
    print(f"  measured [{name}] M={M} K={K} N={N}", flush=True)
    X = torch.randn(M, K, device=dev)
    W = torch.randn(K, N, device=dev) * 0.02
    Wq, _ = sr_kernel.quantize_weight(W)
    _ = sr_kernel.qmatmul(X, Wq)


print("=== warmup ===", flush=True)
# ---------- Global warmup: 캡처 안 함 ----------
X_wm = torch.randn(4096, 768, device=dev)
W_wm = torch.randn(768, 768, device=dev) * 0.02
Wq_wm, _ = sr_kernel.quantize_weight(W_wm)                  # +2 matches
for _ in range(3):
    _ = sr_kernel.qmatmul(X_wm, Wq_wm)                      # +3 matches × 3
torch.cuda.synchronize()
print("=== measured ===", flush=True)
# 여기까지 총 11 matching kernels → --launch-skip-before-match 11


# ---------- Measured: GPT-2 small 이 실제 만나는 4개 shape ----------
SHAPES = [
    ("c_attn",     4096,  768, 2304),   # (M, K, N)
    ("c_proj",     4096,  768,  768),
    ("c_fc",       4096,  768, 3072),
    ("mlp.c_proj", 4096, 3072,  768),
]

for name, M, K, N in SHAPES:
    one_pass(name, M, K, N)                                 # +5 matches each
torch.cuda.synchronize()
print("=== done ===", flush=True)
# 4 × 5 = 20 matching kernels → -c 20
