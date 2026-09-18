"""
GPT-2 INT4 fake-quant: PPL, prefill latency, decode latency.

Tensors are FP32; cuBLAS GEMM is TF32 (allow_tf32=True). Config 3 MatMul is FP32 FMA.

  1 TF32     : original Conv1D, cuBLAS TF32 GEMM
  2 ref-quant: PyTorch fake-quant + cuBLAS TF32
  3 tiled    : CUDA fused fake-quant + custom tiled GEMM  (custom_qmatmul)
  4 kernel   : CUDA fused fake-quant + cuBLAS TF32        (qmatmul)

GPT-2 Conv1D.weight is (K, N); K is 768 or 3072 (multiple of GROUP_SIZE=16).
"""
import copy
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers.pytorch_utils import Conv1D

from qmatmul import sr_kernel, ref_fake_quant, bench, GROUP

dev = "cuda"
torch.manual_seed(0)

# Same TF32 flag on every cuBLAS path (configs 1, 2, 4).
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ===============================================================
# CUDA fake-quant
# ===============================================================
class QuantConv1D(nn.Module):
    """Config 4: fused fake-quant + cuBLAS. Weight quantized once at init."""
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


class CustomQuantConv1D(QuantConv1D):
    """Config 3: fused fake-quant + tiled shared-memory GEMM."""
    def forward(self, x):
        return sr_kernel.custom_qmatmul(x, self.wq) + self.bias


# ===============================================================
# PyTorch fake-quant
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


# ===============================================================
# Swap helpers
# ===============================================================
def swap(model, cls, targets=None):
    """Replace Conv1D with cls. Collect names first; do not mutate while iterating."""
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


# ===============================================================
# Perplexity (sliding window, token-weighted)
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
        valid = trg_len - 1  # causal shift drops one token
        nll_sum += loss.item() * valid
        n_tok += valid

        prev_end = end
        if end == seq_len:
            break
    return float(torch.exp(torch.tensor(nll_sum / n_tok)))


def load_wikitext(tok, limit=None):
    from datasets import load_dataset
    # huggingface_hub 1.x rejects the bare "wikitext" id.
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    enc = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids.to(dev)
    if limit:
        enc = enc[:, :limit]
    print(f"wikitext-2 test : {enc.size(1)} tokens"
          + (f" (limit={limit})" if limit else ""))
    return enc


@torch.no_grad()
def bench_decode(model, B, prompt_len=128, warmup=10, iters=50):
    """ms / token. Prefill is outside the timer; each step is M=1 with KV cache."""
    prompt = torch.randint(0, 50257, (B, prompt_len), device=dev)
    out = model(prompt, use_cache=True)
    past = [out.past_key_values]
    token = torch.randint(0, 50257, (B, 1), device=dev)

    def step():
        o = model(token, past_key_values=past[0], use_cache=True)
        past[0] = o.past_key_values

    return bench(step, warmup=warmup, iters=iters)


def _print_row(label, times, unit="ms"):
    t1, t2, t3, t4 = times
    print(f"  {label:<18}  "
          f"{t1:7.2f}{unit} {t2:7.2f}{unit} {t3:7.2f}{unit} {t4:7.2f}{unit}   "
          f"x{t1/t4:>6.2f}  x{t2/t4:>6.2f}  x{t3/t4:>6.2f}")


# ===============================================================
def main(ppl_limit=None):
    print(torch.cuda.get_device_name(0))

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    # Weights stay float32 (kernel rejects fp16). cuBLAS GEMM is TF32.
    # lm_head stays nn.Linear (tied with wte).
    base = GPT2LMHeadModel.from_pretrained("gpt2").to(dev).eval().float()
    ref, _    = build(base, RefQuantConv1D)
    tiled, _  = build(base, CustomQuantConv1D)
    kernel, n = build(base, QuantConv1D)
    print(f"replaced {len(n)} Conv1D modules (configs 2-4)")

    cfgs = [("tf32", base), ("ref-fq", ref), ("tiled", tiled), ("kernel", kernel)]

    print("\n=== Perplexity (wikitext-2) ===")
    enc = load_wikitext(tok, limit=ppl_limit)
    for tag, mdl in cfgs:
        print(f"  {tag:<8} ppl = {perplexity(mdl, tok, enc):8.3f}")

    hdr = (f"  {'':<18}  {'tf32':>8}  {'ref-fq':>8}  {'tiled':>8}  {'kernel':>8}   "
           f"{'#4/#1':>8}  {'#4/#2':>8}  {'#4/#3':>8}")

    print("\n=== Prefill latency ===")
    print(hdr)
    for B, S in [(1, 512), (8, 512), (16, 1024)]:
        ids = torch.randint(0, 50257, (B, S), device=dev)
        with torch.no_grad():
            times = [bench(lambda m=mdl: m(ids), warmup=3, iters=10) for _, mdl in cfgs]
        _print_row(f"B={B:<2} S={S:<4} M={B*S:<5}", times)

    print("\n=== Decode latency (ms/token, KV cache, M=1) ===")
    print(hdr)
    for B in [1, 8]:
        times = [bench_decode(mdl, B) for _, mdl in cfgs]
        _print_row(f"B={B:<2} prompt=128", times)

if __name__ == "__main__":
    # ref-quant is slow (amax/fp8 every forward). Use a limit for a smoke run;
    # set ppl_limit=None for the reported number.
    main(ppl_limit=50_000)
