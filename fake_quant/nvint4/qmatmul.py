"""
CUDA fake-quant + matmul 커널의 Python 래퍼.

- sr_kernel      : JIT 빌드된 확장 모듈 (quantize_weight / qmatmul / ...)
- ref_fake_quant : 커널과 동일한 연산의 순수 PyTorch 구현 (정확도 레퍼런스)
- bench          : CUDA event 기반 시간 측정
"""
import os
import torch
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))

# --use_fast_math 는 나눗셈을 근사 명령으로 바꿔 레퍼런스와 미세하게
# 달라질 수 있다. 정확도 문제가 의심되면 이 플래그부터 빼고 재시도할 것.
_FAST_MATH = False

sr_kernel = load(
    name="sr_kernel",
    sources=[os.path.join(_HERE, "kernel.cu"),
             os.path.join(_HERE, "binding.cpp")],
    extra_cuda_cflags=["-O3"] + (["--use_fast_math"] if _FAST_MATH else []),
    extra_cflags=["-O3"],
    verbose=True,
)

GROUP    = 16
QMAX     = 7
E4M3_MAX = 448.0


# ---------------------------------------------------------------
# PyTorch reference : 커널과 같은 순서로 계산해야 의미가 있다
# ---------------------------------------------------------------
def ref_fake_quant(x: torch.Tensor, group_dim: int):
    """
    커널의 FakeQuantColDir / FakeQuantRowDir 과 동일한 연산.

    group_dim : 그룹을 묶는 축 (= K 축)
        activation (M, K) -> group_dim = -1
        weight     (K, N) -> group_dim =  0

    반환 : (fake-quant 된 x, s_g)
        s_g 는 group 축을 제외한 shape. weight 의 경우 (N, K/GROUP) 이므로
        커널 출력 (K/GROUP, N) 과 비교하려면 .t() 가 필요하다.
    """
    t_max = x.abs().max()
    s_t = (t_max / E4M3_MAX) / QMAX
    if s_t.item() == 0.0:
        s_t = torch.ones_like(s_t)

    xm = x.movedim(group_dim, -1)                 # 그룹 축을 마지막으로
    shape = xm.shape
    assert shape[-1] % GROUP == 0, f"group dim {shape[-1]} not divisible by {GROUP}"
    xg = xm.reshape(*shape[:-1], shape[-1] // GROUP, GROUP)

    g_max = xg.abs().amax(-1, keepdim=True)

    # fp8 E4M3 라운딩을 반드시 재현해야 한다.
    # s_g_raw = g_max * E4M3_MAX / t_max <= E4M3_MAX 이므로 overflow 없음.
    s_g = ((g_max / QMAX) / s_t).to(torch.float8_e4m3fn).float()

    s = s_g * s_t
    s = torch.where(s == 0, torch.ones_like(s), s)

    q = torch.round(xg / s).clamp(-QMAX, QMAX)
    out = (q * s).reshape(shape).movedim(-1, group_dim).contiguous()
    return out, s_g.squeeze(-1)


# ---------------------------------------------------------------
# benchmark helper
# ---------------------------------------------------------------
def bench(fn, warmup=20, iters=100):
    """CUDA event 로 평균 실행시간(ms)을 측정. warmup 은 반드시 필요하다."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters