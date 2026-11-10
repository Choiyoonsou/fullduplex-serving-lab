"""KV cache policies (B0/B1/B2/P1/P2) expressed two equivalent ways.

1. `attention_mask(policy, keep)`: a [S, S] boolean mask for a single *parallel*
   teacher-forced forward. Because Moshi stores keys *after* RoPE (absolute
   positions), "which past frames are still in the cache at step s" fully
   determines the streaming output. So a ring buffer / sink / skip-write policy
   is exactly a sparsity pattern on the causal mask. This turns ~S sequential
   streaming steps into one forward pass (the main efficiency trick).

2. `PolicyKVCache`: a drop-in replacement for moshi's `RingKVCache` that
   implements the same policies step by step (for streaming verification and
   for the later real multi-session serving measurement on the 3090).

Semantics shared by both (budget C = number of KV slots, incl. current frame):
  * At step s the query sees: the current frame s (always), the first `n_sink`
    *committed* frames, and the most recent `C - 1 - n_sink` committed frames.
  * A frame is committed (its KV written) iff keep[s] (P1/P2) or always (B*).
  * A non-committed frame is still computed with its own query; later frames
    just cannot see it.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

MOSHI_CONTEXT = 3000          # moshi temporal transformer context (frames)
N_LAYERS, N_HEADS, HEAD_DIM = 32, 32, 128


@dataclass(frozen=True)
class Policy:
    kind: str                    # full | B0 | B1 | B2 | P1 | P2
    slots: int = MOSHI_CONTEXT   # KV slots per layer (incl. current frame)
    n_sink: int = 0              # B1 attention-sink frames
    skip_silence: bool = False   # P1/P2
    kv_bits: int | None = None   # B2/P2: 8 or 4 (None = fp16/bf16)
    recent: int = 1              # last `recent` frames (incl. current) always kept, silent or not
    rvq_k: int = 8               # Moshi-channel codebooks present in the *input* (RVQ step control); 8 = all

    @property
    def tag(self) -> str:
        t = f"{self.kind}_s{self.slots}"
        if self.n_sink:
            t += f"_sink{self.n_sink}"
        if self.recent > 1:
            t += f"_w{self.recent}"
        if self.kv_bits:
            t += f"_int{self.kv_bits}"
        if self.rvq_k < 8:
            t += f"_rvq{self.rvq_k}"
        return t

    def bytes_per_slot(self) -> float:
        """KV bytes per frame slot across all layers (K and V), incl. scales."""
        per_head = HEAD_DIM * 2 if self.kv_bits is None else HEAD_DIM * self.kv_bits / 8 + 2
        return N_LAYERS * 2 * N_HEADS * per_head

    def kv_mb(self) -> float:
        return self.slots * self.bytes_per_slot() / 2**20

    def bf16_equiv_frames(self) -> float:
        """Memory budget expressed in bf16 frames (x-axis of the main plot)."""
        return self.slots * self.bytes_per_slot() / (N_LAYERS * 2 * N_HEADS * HEAD_DIM * 2)


KV_KINDS = ("B0", "B1", "P1", "P1S", "P1W", "B2", "P2")


def make_policies(budgets=(125, 250, 500, 1000), n_sink=4, int4=False, kinds=KV_KINDS) -> list[Policy]:
    """Default grid. `budget` = memory in bf16 frames; int8 policies get 2x slots.

    02_verify showed Moshi collapses without attention sinks (B0 budget 100:
    text NLL 2.63 vs 0.46 full; B1 with 4 sink frames: 0.48). So the fair test of
    the silence idea is P1S (skip-silence + sink) vs B1; P1 (no sink) vs B0 is
    kept for the plan's original comparison. P1S also hurt audio (cb0) NLL at
    every budget because it drops the *recent* silent frames; P1W keeps a local
    window of the last budget/4 frames (silent or not) and applies skip-silence
    only to older memory. B2 = B1 + int8, P2 = P1W + int8 (same window in time).
    """
    ps = [Policy("full"), Policy("full", kv_bits=8)]
    if int4:
        ps.append(Policy("full", kv_bits=4))
    for c in budgets:
        ps.append(Policy("B0", c))
        ps.append(Policy("B1", c, n_sink=n_sink))
        ps.append(Policy("P1", c, skip_silence=True))
        ps.append(Policy("P1S", c, n_sink=n_sink, skip_silence=True))
        ps.append(Policy("P1W", c, n_sink=n_sink, skip_silence=True, recent=c // 4))
        ps.append(Policy("B2", 2 * c, n_sink=n_sink, kv_bits=8))
        ps.append(Policy("P2", 2 * c, n_sink=n_sink, skip_silence=True, kv_bits=8, recent=c // 2))
        if int4:
            ps.append(Policy("B2", 4 * c, n_sink=n_sink, kv_bits=4))
            ps.append(Policy("P2", 4 * c, n_sink=n_sink, skip_silence=True, kv_bits=4, recent=c))
    return [p for p in ps if p.kind == "full" or p.kind in kinds]


def variant_policies(rvq_ks=(), kv_bits=()) -> list[Policy]:
    """Full-context variants for the RVQ and KV-quantization experiments (reference = Policy("full"))."""
    ps = [Policy("full")]
    ps += [Policy("full", rvq_k=k) for k in sorted(set(rvq_ks), reverse=True) if k < 8]
    ps += [Policy("full", kv_bits=b) for b in sorted(set(kv_bits), reverse=True)]
    return ps


def attention_mask(policy: Policy, keep: torch.Tensor) -> torch.Tensor:
    """keep: [S] bool (per transformer input step). Returns [S, S] bool (True = attend)."""
    S = keep.shape[0]
    dev = keep.device
    if not policy.skip_silence:
        keep = torch.ones_like(keep)
    keep = keep.bool()
    C = min(policy.slots, MOSHI_CONTEXT)
    W = policy.recent
    n_ring = C - W - policy.n_sink
    assert n_ring >= 1 and W >= 1, policy
    q = torch.arange(S, device=dev)[:, None]
    k = torch.arange(S, device=dev)[None, :]
    recent = (k <= q) & (q - k < W)
    cnt = torch.cumsum(keep.long(), 0)                      # kept in [0, j]
    # kept in [0, s-W] (frames that have left the recent window)
    idx = torch.arange(S, device=dev) - W
    cnt_left = torch.where(idx >= 0, cnt[idx.clamp(min=0)], cnt.new_zeros(()))
    old_kept = (q - k >= W) & keep[None, :]
    is_sink = cnt[None, :] <= policy.n_sink
    newer = cnt_left[:, None] - cnt[None, :]                # kept frames in (j, s-W]
    vis = old_kept & (is_sink | (newer < n_ring))
    return vis | recent


def full_mask(S: int, device) -> torch.Tensor:
    q = torch.arange(S, device=device)[:, None]
    k = torch.arange(S, device=device)[None, :]
    return (k <= q) & (q - k < MOSHI_CONTEXT)


def fake_quant(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric absmax quant-dequant per (token, head) over head_dim. x: [..., D]."""
    qmax = 2 ** (bits - 1) - 1
    xf = x.float()
    scale = xf.abs().amax(-1, keepdim=True).clamp(min=1e-8) / qmax
    scale = scale.to(torch.float16).float()            # scales stored as fp16
    return (xf / scale).round().clamp(-qmax - 1, qmax).mul(scale).to(x.dtype)


def keep_from_speech(speech: torch.Tensor, hang: int = 1) -> torch.Tensor:
    """speech: [2, F] bool per Mimi frame (ch0=Moshi, ch1=user).
    Returns keep: [F] bool per transformer *input step*.

    Input step s carries text/semantic of frame s-1 and acoustic codes of
    frame s-2 (moshi delays), so step s is "speech" if frame s-1 or s-2 is.
    `hang` dilates speech regions by that many frames on both sides so turn
    boundaries (onset/offset cues) are kept. Step 0 (initial token) is kept.
    """
    sp = speech.any(0)
    F = sp.shape[0]
    if hang > 0:
        sp = torch.nn.functional.max_pool1d(sp.float()[None, None], 2 * hang + 1, 1, hang)[0, 0] > 0
    keep = torch.zeros(F, dtype=torch.bool, device=speech.device)
    keep[1:] |= sp[:-1]
    keep[2:] |= sp[:-2]
    keep[0] = True
    return keep


class PolicyKVCache:
    """Streaming KV cache implementing the same policies as `attention_mask`.

    Interface-compatible with moshi.modules.transformer.RingKVCache.complete()
    for T=1 steps. Physical slots (= policy.slots):
        [ n_sink sink | n_ring long-term ring | W-1 recent ring | 1 scratch ]
    The current frame is written to scratch. At the start of the next step it
    is pushed into the recent ring; the frame falling out of the recent ring
    goes to long-term memory (sink first, then ring) only if it was kept
    (B*: always; P*: speech). With W=1 frames go straight from scratch to
    long-term. Deferring the push keeps the to-be-evicted entry visible during
    the current step, matching moshi's ring semantics.
    `keep_ref.keep` ([B] bool) must be set before each step (P1/P2).
    Returned positions are absolute frame indices (-1 = empty), so the caller
    must disable moshi's `delta < context` check (mha.context = None): with
    skip-write, retained entries can be older than `slots` frames.
    """

    def __init__(self, B, H, D, policy: Policy, keep_ref, device, dtype):
        self.p = policy
        self.n_sink, self.W = policy.n_sink, policy.recent
        self.n_ring = policy.slots - self.W - self.n_sink
        assert self.n_ring >= 1 and self.W >= 1
        self.n_rec = self.W - 1
        self.rec0 = self.n_sink + self.n_ring           # first recent slot
        self.n_slots = policy.slots
        self.scratch = self.n_slots - 1
        self.bits, self.dtype, self.keep_ref = policy.kv_bits, dtype, keep_ref
        store_dtype = torch.int8 if self.bits else dtype
        self.cache = torch.zeros((2, B, H, self.n_slots, D), device=device, dtype=store_dtype)
        self.scale = (torch.ones((2, B, H, self.n_slots, 1), device=device, dtype=torch.float16)
                      if self.bits else None)
        self.pos = torch.full((B, self.n_slots), -1, device=device, dtype=torch.long)
        self.slot_keep = torch.zeros((B, self.n_slots), device=device, dtype=torch.bool)
        self.n_committed = torch.zeros(B, device=device, dtype=torch.long)
        self.n_pushed = torch.zeros(B, device=device, dtype=torch.long)   # frames pushed to recent ring
        self.t = torch.zeros(B, device=device, dtype=torch.long)
        self.has_pending = False

    def reset(self, reset_mask):
        for x in (self.pos,):
            x[reset_mask] = -1
        self.slot_keep[reset_mask] = False
        for x in (self.n_committed, self.n_pushed, self.t):
            x[reset_mask] = 0

    def _bufs(self):
        return [self.cache] + ([self.scale] if self.scale is not None else [])

    def _move(self, src, dst, cond):
        """Per batch b: copy slot src[b] -> dst[b] where cond[b]. src/dst/cond: [B]."""
        _, B, H, _, _ = self.cache.shape
        for buf in self._bufs():
            d = buf.shape[-1]
            gs = src.view(1, B, 1, 1, 1).expand(2, B, H, 1, d)
            gd = dst.view(1, B, 1, 1, 1).expand(2, B, H, 1, d)
            new, old = buf.gather(3, gs), buf.gather(3, gd)
            buf.scatter_(3, gd, torch.where(cond.view(1, B, 1, 1, 1), new, old))
        for meta in (self.pos, self.slot_keep):
            new, old = meta.gather(1, src[:, None]), meta.gather(1, dst[:, None])
            meta.scatter_(1, dst[:, None], torch.where(cond[:, None], new, old))

    def _to_long_term(self, src, cond):
        nc = self.n_committed
        dst = torch.where(nc < self.n_sink, nc, self.n_sink + (nc - self.n_sink) % self.n_ring)
        self._move(src, dst, cond)
        self.n_committed += cond.long()

    def _push(self):
        B = self.pos.shape[0]
        scratch = torch.full((B,), self.scratch, device=self.pos.device, dtype=torch.long)
        if self.n_rec == 0:
            self._to_long_term(scratch, self.slot_keep[:, self.scratch])
            return
        dst = self.rec0 + self.n_pushed % self.n_rec
        full = self.n_pushed >= self.n_rec              # dst holds the oldest recent frame
        evict_keep = full & self.slot_keep.gather(1, dst[:, None])[:, 0]
        self._to_long_term(dst, evict_keep)
        self._move(scratch, dst, torch.ones_like(full))
        self.n_pushed += 1

    def complete(self, k, v, exec_mask):
        from moshi.modules.transformer import KVCacheResult
        B, H, T, D = k.shape
        assert T == 1, "PolicyKVCache supports single-step streaming only"
        if self.has_pending:
            self._push()
        sl = slice(self.scratch, self.scratch + 1)
        if self.bits:
            qmax = 2 ** (self.bits - 1) - 1
            for i, x in enumerate((k, v)):
                xf = x.float()
                sc = (xf.abs().amax(-1, keepdim=True).clamp(min=1e-8) / qmax).to(torch.float16)
                self.cache[i, :, :, sl] = (xf / sc.float()).round().clamp(-qmax - 1, qmax).to(torch.int8)
                self.scale[i, :, :, sl] = sc
            kv = (self.cache.float() * self.scale.float()).to(self.dtype)
            keys, values = kv[0], kv[1]
        else:
            self.cache[0, :, :, sl] = k
            self.cache[1, :, :, sl] = v
            keys, values = self.cache[0], self.cache[1]
        self.pos[:, self.scratch] = self.t
        keep = self.keep_ref.keep if self.p.skip_silence else None
        self.slot_keep[:, self.scratch] = True if keep is None else keep.to(self.pos.device).bool()
        self.has_pending = True
        positions = self.pos.clone()
        self.t += 1
        return KVCacheResult(keys, values, positions)
