import torch
from qmatmul import sr_kernel, ref_fake_quant, GROUP

torch.manual_seed(0)
dev = "cuda"

def report(name, a, b):
    print(f'  {name:<14} kernel is not zero: {a.ne(0).any()}, PyTorch is not zero: {b.ne(0).any()}')
    diff = (a - b).abs()
    denom = b.abs().clamp_min(1e-12)
    rel = (diff / denom)[b.abs() > 1e-6]
    bad = (diff > 1e-5).sum().item()
    print(f"  {name:<14} max_abs={diff.max():.3e}  "
          f"max_rel={rel.max() if rel.numel() else 0:.3e}  "
          f"mismatch={bad}/{a.numel()}")


def run(M, K, N, tag):
    print(f"\n=== {tag}  M={M} K={K} N={N} ===")
    X = torch.randn(M, K, device=dev) * 0.5
    W = torch.randn(K, N, device=dev) * 0.02

    # --- ① weight fake-quant ---
    Wq, sw = sr_kernel.quantize_weight(W)
    Wq_ref, sw_ref = ref_fake_quant(W, group_dim=0)   # W는 K가 dim 0
    report("Wq", Wq, Wq_ref)
    report("scale_w", sw, sw_ref.t().contiguous())    # ref는 (N,K/16)

    # --- ② activation + matmul ---
    C = sr_kernel.qmatmul(X, Wq)
    Xq_ref, _ = ref_fake_quant(X, group_dim=-1)       # X는 K가 마지막
    C_ref = Xq_ref @ Wq_ref
    report("C", C, C_ref)

    # --- fp32 baseline 대비 (양자화 손실 확인용) ---
    C_fp32 = X @ W
    cos = torch.nn.functional.cosine_similarity(
        C.flatten(), C_fp32.flatten(), dim=0).item()
    err = (C - C_fp32).norm() / C_fp32.norm()
    print(f"  vs torch.matmul: cos={cos:.6f}  rel_fro={err:.4f}")


# GPT-2 small의 실제 shape
run(64, 768, 2304, "c_attn")
run(64, 768,  768, "c_proj")
run(64, 768, 3072, "c_fc")
run(64, 3072, 768, "mlp.c_proj")

# 3D 입력 (batch, seq, hidden) 경로
print("\n=== 3D shape ===")
X3 = torch.randn(2, 32, 768, device=dev) * 0.5
W = torch.randn(768, 3072, device=dev) * 0.02
Wq, _ = sr_kernel.quantize_weight(W)
C3 = sr_kernel.qmatmul(X3, Wq)
print(f"  out shape {tuple(C3.shape)}  expected (2, 32, 3072)")
assert C3.shape == (2, 32, 3072)

# 같은 입력 두 번 → tmax 초기화 검증
print("\n=== tmax 재현성 ===")
X = torch.randn(64, 768, device=dev)
c1, c2 = sr_kernel.qmatmul(X, Wq), sr_kernel.qmatmul(X, Wq)
print(f"  identical: {torch.equal(c1, c2)}")