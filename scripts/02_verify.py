"""Correctness gate before running the grid (risk table: "P1 구현 후 출력이 깨짐").

 (1) patched masked forward (policy=full)  == unpatched moshi forward
 (2) masked B0 forward                     == real moshi streaming with RingKVCache(capacity=C)
 (3) masked B1/P1/P2 forward               == streaming with PolicyKVCache
 (4) fp16 (T4) vs bf16 NLL on full context (needs 2 more GPUs; --bf16_gpus)

  python scripts/02_verify.py --data data/prep_test/r0.6 --steps 400 --gpus 0 1 --bf16_gpus 2 3
"""
import argparse
import glob
import os
import sys

os.environ.setdefault("HF_HOME", os.path.abspath("hf_cache"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from fdkv.model import CTX, delayed_inputs, load_moshi, per_frame_nll, teacher_forced, temporal_layers  # noqa: E402
from fdkv.policies import Policy, PolicyKVCache, attention_mask, full_mask, keep_from_speech  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/prep_test/r0.6")
ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--budget", type=int, default=100)
ap.add_argument("--gpus", type=int, nargs=2, default=[0, 1])
ap.add_argument("--bf16_gpus", type=int, nargs=2, default=None)
args = ap.parse_args()

lm, _ = load_moshi(f"cuda:{args.gpus[0]}", f"cuda:{args.gpus[1]}", torch.float16)
d = torch.load(sorted(glob.glob(f"{args.data}/*.pt"))[0])
S = args.steps
codes = d["codes"][None, :, :S].long()
keep = keep_from_speech(d["speech"][:, :S])
print(f"dialog {d['id']} ratio {d['ratio']:.2f}, steps {S}, keep frac {keep.float().mean():.2f}")


def masked_text_logits(policy):
    CTX.set(attention_mask(policy, keep)[None], [policy.kv_bits])
    r = teacher_forced(lm, codes, need_audio=False)
    CTX.clear()
    return r["text_logits"][0].float()


class KeepRef:
    keep = None


def streaming_text_logits(policy, use_moshi_ring=False):
    delayed = delayed_inputs(lm, codes.to(lm.dev_a))[:, :, :-1]
    mhas = temporal_layers(lm)
    old_ctx = [m.context for m in mhas]
    if use_moshi_ring:
        for m in mhas:
            m.context = policy.slots
    ref, outs = KeepRef(), []
    with lm.transformer.streaming(1):
        if not use_moshi_ring:
            for m in mhas:
                st = m._streaming_state
                st.kv_cache = PolicyKVCache(1, m.num_heads, m.embed_dim // m.num_heads, policy, ref,
                                            st.kv_cache.cache.device, st.kv_cache.cache.dtype)
                m.context = None   # positions are absolute now (the "delta < context" fix)
        for s in range(S):
            ref.keep = keep[s:s + 1].to(lm.dev_a)
            _, tl = lm.forward_text(delayed[:, :, s:s + 1])
            outs.append(tl[0, 0, 0].float())
    for m, c in zip(mhas, old_ctx):
        m.context = c
    return torch.stack(outs)


def compare(name, a, b):
    tgt = codes[0, 0].to(a.device).clamp(min=0)
    na = F.cross_entropy(a, tgt, reduction="none")
    nb = F.cross_entropy(b.to(a.device), tgt, reduction="none")
    diff = (a - b.to(a.device)).abs()
    print(f"  {name:38s} max|dlogit| {diff.max():.4f}  mean|dlogit| {diff.mean():.5f}  "
          f"NLL {na.mean():.4f} vs {nb.mean():.4f}  max|dNLL| {(na - nb).abs().max():.4f}")
    return (na - nb).abs().mean().item()


with torch.no_grad():
    print("(1) patched full vs original moshi forward")
    ref_full = masked_text_logits(Policy("full"))
    orig = teacher_forced(lm, codes, need_audio=False)["text_logits"][0].float()
    compare("full(mask) vs moshi", ref_full, orig)

    C = args.budget
    print(f"(2)(3) parallel mask vs step-by-step streaming, budget {C} (noise floor = row 'full')")
    compare("full: mask vs moshi streaming(3000)", ref_full, streaming_text_logits(Policy("B0", 3000), True))
    for p, ring in [(Policy("B0", C), True), (Policy("B0", C), False), (Policy("B1", C, n_sink=4), False),
                    (Policy("P1", C, skip_silence=True), False),
                   (Policy("P1W", C, n_sink=4, skip_silence=True, recent=C // 4), False),
                   (Policy("P2", 2 * C, n_sink=4, skip_silence=True, kv_bits=8, recent=C // 2), False)]:
        compare(f"{p.tag}: mask vs {'moshi Ring' if ring else 'PolicyKVCache'}",
                masked_text_logits(p), streaming_text_logits(p, ring))
    # sanity: budget actually changes things
    compare(f"sanity: full vs B0_{C} (should differ)", ref_full, masked_text_logits(Policy("B0", C)))

if args.bf16_gpus:
    print("(4) fp16 vs bf16, full context, all frames of up to 3 dialogs")
    lm16 = lm
    lmbf, _ = load_moshi(f"cuda:{args.bf16_gpus[0]}", f"cuda:{args.bf16_gpus[1]}", torch.bfloat16)
    for f in sorted(glob.glob(f"{args.data}/*.pt"))[:3]:
        dd = torch.load(f)
        c = dd["codes"][None].long()
        res = []
        for m in (lm16, lmbf):
            CTX.set(full_mask(c.shape[-1], "cpu")[None], [None])
            n = per_frame_nll(teacher_forced(m, c), c)
            CTX.clear()
            res.append(n)
        t16, tbf = np.nanmean(res[0]["text"]), np.nanmean(res[1]["text"])
        a16, abf = np.nanmean(res[0]["audio"][:, 0]), np.nanmean(res[1]["audio"][:, 0])
        print(f"  {dd['id']}: text NLL fp16 {t16:.4f} bf16 {tbf:.4f} | cb0 NLL fp16 {a16:.4f} bf16 {abf:.4f} "
              f"| nan/inf fp16: {np.isnan(res[0]['text']).sum() - np.isnan(res[1]['text']).sum()}")
