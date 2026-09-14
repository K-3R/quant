"""
GPT-2 에 커널을 이식하고 3-way 로 비교한다.

  fp32       : original
  ref-quant  : fake-quant using PyTorch
  kernel     : CUDA kernel

  fp32 vs ref-quant  차이 = int4 + fp8 group scale 이 원래 내는 손실
  ref-quant vs kernel 차이 = 0 에 가까워야 정상

GPT-2
  - Conv1D.weight : (K, N)
    (nn.Linear weight : (N, K) -> Need to transpose)
  - K 가 768 / 3072 로 전부 GROUP_SIZE(16)의 배수.
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers.pytorch_utils import Conv1D

from qmatmul import sr_kernel, ref_fake_quant, bench, GROUP

dev = "cuda"
torch.manual_seed(0)

# fp32 / ref-quant / kernel 세 경로 모두 같은 cuBLAS 백엔드를 쓰므로
# TF32 도 세 곳에 동시에 적용된다 (공정 비교 유지).
# int4 + fp8 group-scale fake-quant 오차 대비 TF32 (mantissa 10bit) 은 무시할 수준.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ===============================================================
# CUDA Fake Quantization
# ===============================================================
class QuantConv1D(nn.Module):
    """W is quantized only in __init__."""
    def __init__(self, orig: Conv1D):
        super().__init__()
        self.nf = orig.nf
        K = orig.weight.size(0)
        assert K % GROUP == 0, f"K={K} not divisible by {GROUP}"
        wq, sw = sr_kernel.quantize_weight(orig.weight.data.contiguous())
        self.register_buffer("wq", wq)
        self.register_buffer("scale_w", sw)
        self.register_buffer("bias", orig.bias.data.clone())

    def forward(self, x):
        return sr_kernel.qmatmul(x, self.wq) + self.bias

# ===============================================================
# PyTorch Fake Quantization
# ===============================================================
class RefQuantConv1D(nn.Module):
    def __init__(self, orig: Conv1D):
        super().__init__()
        self.nf = orig.nf
        wq, _ = ref_fake_quant(orig.weight.data.contiguous(), group_dim=0)
        self.register_buffer("wq", wq)
        self.register_buffer("bias", orig.bias.data.clone())

    def forward(self, x):
        shape = x.shape[:-1] + (self.nf,)
        x2 = x.reshape(-1, x.size(-1))
        xq, _ = ref_fake_quant(x2, group_dim=-1)
        return (xq @ self.wq).reshape(shape) + self.bias


class WeightOnlyConv1D(RefQuantConv1D):
    """weight 만 양자화 -- 손실의 출처를 가리기 위한 ablation."""
    def forward(self, x):
        shape = x.shape[:-1] + (self.nf,)
        return (x.reshape(-1, x.size(-1)) @ self.wq).reshape(shape) + self.bias


class ActOnlyConv1D(nn.Module):
    """activation 만 양자화."""
    def __init__(self, orig: Conv1D):
        super().__init__()
        self.nf = orig.nf
        self.register_buffer("w", orig.weight.data.clone())
        self.register_buffer("bias", orig.bias.data.clone())

    def forward(self, x):
        shape = x.shape[:-1] + (self.nf,)
        xq, _ = ref_fake_quant(x.reshape(-1, x.size(-1)), group_dim=-1)
        return (xq @ self.w).reshape(shape) + self.bias


# ===============================================================
# 교체 유틸
# ===============================================================
def swap(model, cls, targets=None):
    """Conv1D 를 cls 로 교체. targets=None 이면 전부.

    named_modules() 를 순회하는 중에 트리를 수정하면 안 되므로
    먼저 리스트로 모은 뒤에 바꾼다.
    """
    todo = [(n, m) for n, m in model.named_modules()
            if isinstance(m, Conv1D) and (targets is None or n in targets)]
    for name, mod in todo:
        parent_name, attr = name.rsplit(".", 1)
        setattr(model.get_submodule(parent_name), attr, cls(mod).to(dev))
    return [n for n, _ in todo]


def build(ref_model, cls, targets=None):
    m = copy.deepcopy(ref_model)
    names = swap(m, cls, targets)
    return m, names


def grab(model, name, ids):
    """특정 모듈의 출력을 hook 으로 뽑아낸다."""
    out = {}
    h = model.get_submodule(name).register_forward_hook(
        lambda mod, inp, o: out.update(y=o.detach()))
    with torch.no_grad():
        model(ids)
    h.remove()
    return out["y"]


def compare(a, b, tag):
    rel = ((a - b).norm() / b.norm()).item()
    cos = F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    print(f"  {tag:<34} rel_fro={rel:.5f}  cos={cos:.6f}")
    return rel, cos


# ===============================================================
# Perplexity (sliding window, token 가중 평균)
# ===============================================================
@torch.no_grad()
def perplexity(model, tok, enc, stride=512, maxlen=1024):
    seq_len = enc.size(1)
    nll_sum, n_tok, prev_end = 0.0, 0, 0
    for begin in range(0, seq_len, stride):
        end = min(begin + maxlen, seq_len)
        trg_len = end - prev_end
        ids = enc[:, begin:end]
        tgt = ids.clone()
        tgt[:, :-trg_len] = -100

        loss = model(ids, labels=tgt).loss
        valid = trg_len - 1                      # shift 때문에 1 줄어듦
        nll_sum += loss.item() * valid
        n_tok += valid

        prev_end = end
        if end == seq_len:
            break
    return float(torch.exp(torch.tensor(nll_sum / n_tok)))


def load_wikitext(tok, limit=None):
    from datasets import load_dataset
    # 네임스페이스 없는 "wikitext" 는 huggingface_hub 1.x 에서 거부된다.
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    enc = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids.to(dev)
    if limit:
        enc = enc[:, :limit]
    print(f"wikitext-2 test : {enc.size(1)} tokens"
          + (f" (limit={limit})" if limit else ""))
    return enc


# ===============================================================
def main(ppl_limit=None):
    print(torch.cuda.get_device_name(0))

    # ---- Step 1. baseline ----
    # 커널이 float32 만 받으므로 fp16 로 로드하면 안 된다.
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    ref_model = GPT2LMHeadModel.from_pretrained("gpt2").to(dev).eval().float()

    ids = tok("The capital of France is", return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        base_logits = ref_model(ids).logits
    print(f"\n[baseline] 예측: '{tok.decode(base_logits[0, -1].argmax())}'")

    # ---- Step 2. 한 레이어만 ----
    print("\n=== Step 2. 단일 레이어 (h.0.mlp.c_fc) ===")
    tgt = "transformer.h.0.mlp.c_fc"
    m_k, _ = build(ref_model, QuantConv1D,    [tgt])
    m_r, _ = build(ref_model, RefQuantConv1D, [tgt])

    y_ref = grab(ref_model, tgt, ids)
    y_k   = grab(m_k, tgt, ids)
    y_r   = grab(m_r, tgt, ids)
    compare(y_r, y_ref, "ref-quant vs fp32")
    compare(y_k, y_ref, "kernel    vs fp32")
    compare(y_k, y_r,   "kernel    vs ref-quant  <- 0 에 가까워야")

    # ---- Step 3. 블록 하나 전체 ----
    print("\n=== Step 3. 블록 0 전체 (4개) ===")
    blk = [f"transformer.h.0.{k}" for k in
           ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]]
    m_k, n_k = build(ref_model, QuantConv1D,    blk)
    m_r, _   = build(ref_model, RefQuantConv1D, blk)
    print(f"  replaced {len(n_k)} modules")
    y_ref = grab(ref_model, "transformer.h.0", ids)[0]
    compare(grab(m_r, "transformer.h.0", ids)[0], y_ref, "ref-quant vs fp32")
    compare(grab(m_k, "transformer.h.0", ids)[0], y_ref, "kernel    vs fp32")

    # ---- Step 4. 전체 교체 ----
    print("\n=== Step 4. 전체 교체 (48개) ===")
    # lm_head 는 nn.Linear 이고 wte 와 weight tying 되어 있으므로 건드리지 않는다.
    qmodel, names = build(ref_model, QuantConv1D)
    rmodel, _     = build(ref_model, RefQuantConv1D)
    print(f"  replaced {len(names)} modules")

    with torch.no_grad():
        lk = qmodel(ids).logits
        lr = rmodel(ids).logits
    compare(lr, base_logits, "ref-quant vs fp32")
    compare(lk, base_logits, "kernel    vs fp32")
    compare(lk, lr,          "kernel    vs ref-quant  <- 1e-3 이하면 OK")

    print(f"  top-1 일치 (kernel vs fp32) : "
          f"{(lk.argmax(-1) == base_logits.argmax(-1)).float().mean():.4f}")
    print(f"  top-1 일치 (kernel vs ref)  : "
          f"{(lk.argmax(-1) == lr.argmax(-1)).float().mean():.4f}")
    print(f"  예측: fp32='{tok.decode(base_logits[0,-1].argmax())}'  "
          f"kernel='{tok.decode(lk[0,-1].argmax())}'")

    # ---- Step 5. Perplexity 3-way ----
    print("\n=== Step 5. Perplexity (wikitext-2) ===")
    enc = load_wikitext(tok, limit=ppl_limit)
    for tag, mdl in [("fp32", ref_model), ("ref-quant", rmodel), ("kernel", qmodel)]:
        print(f"  {tag:<12} ppl = {perplexity(mdl, tok, enc):8.3f}")

    # ---- Step 6. ablation ----
    print("\n=== Step 6. 손실의 출처 (ref 구현으로 ablation) ===")
    for tag, cls in [("weight only", WeightOnlyConv1D),
                     ("act only",    ActOnlyConv1D)]:
        m, _ = build(ref_model, cls)
        print(f"  {tag:<12} ppl = {perplexity(m, tok, enc):8.3f}")
        del m
        torch.cuda.empty_cache()

    print("\n=== Step 6b. 레이어 종류별 (kernel) ===")
    for kind in ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]:
        t = [f"transformer.h.{i}.{kind}" for i in range(12)]
        m, _ = build(ref_model, QuantConv1D, t)
        print(f"  {kind:<12} ppl = {perplexity(m, tok, enc):8.3f}")
        del m
        torch.cuda.empty_cache()

    # ---- Step 7. 속도 ----
    # generate() 로 재면 KV cache 때문에 M=1 이 되어 launch 오버헤드만 보인다.
    # prefill 상황으로 재야 GEMM 성능이 드러난다.
    # fp32   : cuBLAS SGEMM baseline
    # ref-fq : PyTorch fake-quant + cuBLAS SGEMM  (커널이 이겨야 하는 대상)
    # kernel : 커널 fake-quant + custom SGEMM     (naive tile 이라 cuBLAS 엔 밀림)
    print("\n=== Step 7. prefill 속도 ===")
    header = f"  {'B':>2} {'S':>4} {'M':>5}   {'fp32':>8}  {'ref-fq':>8}  {'kernel':>8}   {'ker/fp32':>8}  {'ker/ref':>8}"
    print(header)
    for B, S in [(1, 512), (8, 512), (16, 1024), (32, 1024)]:
        big = torch.randint(0, 50257, (B, S), device=dev)
        with torch.no_grad():
            t_f = bench(lambda: ref_model(big), warmup=3, iters=10)
            t_r = bench(lambda: rmodel(big),    warmup=3, iters=10)
            t_k = bench(lambda: qmodel(big),    warmup=3, iters=10)
        print(f"  {B:>2} {S:>4} {B*S:>5}   "
              f"{t_f:7.2f}ms {t_r:7.2f}ms {t_k:7.2f}ms   "
              f"x{t_f/t_k:>6.2f}  x{t_r/t_k:>6.2f}")

if __name__ == "__main__":
    # ref-quant 는 unfold/amax/fp8 캐스팅이 forward 마다 48번 돌아 많이 느리다.
    # 빠르게 확인하려면 ppl_limit 를 주고, 최종 수치는 None 으로 다시 돌릴 것.
    main(ppl_limit=50_000)
