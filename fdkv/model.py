"""Load Moshi split over 2 GPUs and run a policy-masked teacher-forced forward.

Why split: moshiko fp16 weights are ~15.4GB, a T4 has 15GB. Layers [0, split)
plus input embeddings go on dev_a, the remaining layers, text head and the
depformer on dev_b. On a 24GB 3090 pass the same device twice.

Why fp16 on T4: T4 (sm_75) has no bf16 tensor cores (bf16 matmul ~9x slower).
`scripts/02_verify.py` checks fp16 NLL against bf16.
"""
from __future__ import annotations

import os
import types

import torch
import torch.nn.functional as F
from einops import rearrange

from .policies import fake_quant

HF_REPO = "kyutai/moshiko-pytorch-bf16"


class _Ctx:
    """Global policy context read by the patched temporal attention."""
    mask: torch.Tensor | None = None       # [B, S, S] bool on cpu/any device
    bits: list | None = None               # per-batch kv bits (None/8/4)
    _dev_cache: dict = {}

    @classmethod
    def set(cls, mask, bits):
        cls.mask, cls.bits, cls._dev_cache = mask, bits, {}

    @classmethod
    def clear(cls):
        cls.mask, cls.bits, cls._dev_cache = None, None, {}

    @classmethod
    def mask_on(cls, device):
        if device not in cls._dev_cache:
            cls._dev_cache[device] = cls.mask.to(device)[:, None]   # [B,1,S,S]
        return cls._dev_cache[device]


CTX = _Ctx


def _policy_attn_forward(self, query, key, value):
    """Non-streaming temporal self-attention with an arbitrary policy mask."""
    if self._streaming_state is not None or CTX.mask is None:
        return self._orig_forward(query, key, value)
    B, T = query.shape[:2]
    q, k, v = rearrange(self.in_projs[0](query), "b t (p h d) -> p b h t d", p=3, h=self.num_heads)
    offset = torch.zeros(B, device=q.device, dtype=torch.long)
    q, k = self.rope(q, k, offset, time_before_heads=False)
    if CTX.bits is not None and any(CTX.bits):
        k = torch.stack([fake_quant(k[i], b) if b else k[i] for i, b in enumerate(CTX.bits)])
        v = torch.stack([fake_quant(v[i], b) if b else v[i] for i, b in enumerate(CTX.bits)])
    mask = CTX.mask_on(q.device)
    assert mask.shape[-1] == T, (mask.shape, T)
    x = F.scaled_dot_product_attention(q, k, v, mask)
    x = rearrange(x, "b h t d -> b t (h d)")
    return self.out_projs[0](x)


def _move_inputs_hook(device):
    def hook(module, args, kwargs):
        args = tuple(a.to(device) if torch.is_tensor(a) else a for a in args)
        kwargs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in kwargs.items()}
        return args, kwargs
    return hook


def load_moshi(dev_a="cuda:0", dev_b="cuda:1", dtype=torch.float16, split=16):
    from moshi.models import loaders
    ci = loaders.CheckpointInfo.from_hf_repo(HF_REPO)
    lm = ci.get_moshi(device="cpu", dtype=dtype)
    lm.eval()
    layers = lm.transformer.layers
    for i, layer in enumerate(layers):
        d = dev_a if i < split else dev_b
        layer.to(d)
        layer.register_forward_pre_hook(_move_inputs_hook(d), with_kwargs=True)
        sa = layer.self_attn
        assert len(sa.in_projs) == 1 and sa.kv_repeat == 1 and not sa.weights_per_step
        sa._orig_forward = sa.forward
        sa.forward = types.MethodType(_policy_attn_forward, sa)
    for name, child in lm.named_children():
        if name == "transformer":
            continue
        child.to(dev_a if name in ("emb", "text_emb") else dev_b)
    for p in lm.parameters():
        assert p.device.type == "cuda", "parameter left on cpu"
    lm.dev_a, lm.dev_b = torch.device(dev_a), torch.device(dev_b)
    return lm, ci


def temporal_layers(lm):
    return [layer.self_attn for layer in lm.transformer.layers]


@torch.no_grad()
def delayed_inputs(lm, codes):
    """codes [B,17,T] -> delayed [B,17,T+1] (incl. initial token), as LMModel.forward."""
    from moshi.models.lm_utils import _delay_sequence
    B = codes.shape[0]
    initial = lm._get_initial_token().to(codes.device).expand(B, -1, -1)
    delayed = _delay_sequence(lm.delays, codes, initial)
    return torch.cat([initial, delayed], dim=2)


@torch.no_grad()
def teacher_forced(lm, codes, need_audio=True, codes_in=None):
    """Same math as LMModel.forward, multi-device aware.

    Returns text_logits [B,T,V] (predicting codes[:,0,t]) and audio logits
    [B,8,T,card] (predicting Moshi codebooks codes[:,1:9,t]) with masks.
    Input step s of the temporal transformer == mask row s == target frame s.
    `codes_in` (optional) replaces the *input* codes only, e.g. Moshi codebooks
    k..7 set to zero_token_id to simulate a server that generates k codebooks.
    """
    from moshi.models.lm_utils import _undelay_sequence
    codes = codes.to(lm.dev_a)
    # codes_in: what the model is fed (e.g. RVQ-masked); codes: prediction targets
    delayed = delayed_inputs(lm, codes if codes_in is None else codes_in.to(lm.dev_a))
    out, text_logits = lm.forward_text(delayed[:, :, :-1])
    text_logits, tmask = _undelay_sequence(lm.delays[:1], text_logits, fill_value=float("nan"))
    res = {"text_logits": text_logits[:, 0], "text_mask": tmask[:, 0].to(lm.dev_b)
           & (codes[:, 0] != lm.zero_token_id).to(lm.dev_b)}
    if need_audio:
        logits = lm.forward_depformer_training(delayed[:, :, 1:].to(lm.dev_b), out)
        logits, amask = _undelay_sequence(lm.delays[lm.audio_offset:lm.audio_offset + lm.dep_q],
                                          logits, fill_value=float("nan"))
        res["audio_logits"] = logits
        res["audio_mask"] = amask & (codes[:, 1:1 + lm.dep_q] != lm.zero_token_id).to(lm.dev_b)
    return res


def per_frame_nll(res, codes, chunk=512):
    """Returns dict of float32 numpy arrays: text [B,T], audio [B,8,T]; NaN where invalid."""
    codes = codes.to(res["text_logits"].device)
    tl = res["text_logits"]
    B, T, V = tl.shape
    tgt = codes[:, 0].clamp(min=0)
    text = torch.empty(B, T, device=tl.device)
    for s in range(0, T, chunk):
        text[:, s:s + chunk] = F.cross_entropy(tl[:, s:s + chunk].float().transpose(1, 2),
                                               tgt[:, s:s + chunk], reduction="none")
    text[~res["text_mask"]] = float("nan")
    out = {"text": text.cpu().numpy()}
    if "audio_logits" in res:
        al = res["audio_logits"]
        atgt = codes[:, 1:9].clamp(min=0)
        audio = torch.empty(B, 8, T, device=al.device)
        for s in range(0, T, chunk):
            audio[:, :, s:s + chunk] = F.cross_entropy(
                al[:, :, s:s + chunk].float().permute(0, 3, 1, 2), atgt[:, :, s:s + chunk], reduction="none")
        audio[~res["audio_mask"]] = float("nan")
        out["audio"] = audio.cpu().numpy()
    return out
