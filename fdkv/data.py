"""DailyTalkContiguous -> 2-channel Moshi token streams with a controlled silence ratio.

Source: kyutai/DailyTalkContiguous (CC-BY-SA 4.0). Stereo 44.1kHz,
ch0 = SPEAKER_MAIN (-> Moshi channel, has word timestamps), ch1 = other
speaker (-> user channel, no words: Moshi's text stream only carries Moshi).

Silence control: speech "blocks" (union of both channels' VAD) are kept
intact; only the gaps *between* blocks are rescaled by a common factor
(preserves the natural rhythm) until the fraction of 80ms frames where both
channels are silent hits the target.
"""
from __future__ import annotations

import json
import math
from collections import deque

import numpy as np
import torch

SR_SRC = 44100
SR_MIMI = 24000
FRAME_RATE = 12.5
VAD_HOP_S = 0.01           # 10ms resolution for segmentation
VAD_DB = -50.0             # recordings have digital silence between turns
FRAME_VAD_DB = -45.0       # 80ms frame speech threshold (both prep & keep mask)
MERGE_GAP_S = 0.3          # pauses shorter than this stay inside a block
MIN_GAP_S = 0.08
BLOCK_PAD_S = 0.1
LEAD_S, TAIL_S = 0.3, 0.5


def _db(x, hop):
    n = len(x) // hop
    e = np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12)
    return 20 * np.log10(e + 1e-9)


def _segments(sp: np.ndarray):
    """bool array at VAD_HOP_S -> list of (start_s, end_s), pauses < MERGE_GAP_S merged."""
    idx = np.flatnonzero(sp)
    if len(idx) == 0:
        return []
    segs, s, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if (i - prev) * VAD_HOP_S > MERGE_GAP_S:
            segs.append((s * VAD_HOP_S, (prev + 1) * VAD_HOP_S))
            s = i
        prev = i
    segs.append((s * VAD_HOP_S, (prev + 1) * VAD_HOP_S))
    return segs


def speech_blocks(wav: np.ndarray, sr: int):
    """wav [N, 2] -> list of (start_s, end_s, first_ch, last_ch).

    Per-channel VAD segments, sorted by time; overlapping segments (overlaps,
    backchannels) are merged into one block that is never cut.
    """
    hop = int(sr * VAD_HOP_S)
    segs = []
    for c in range(2):
        segs += [(s, e, c) for s, e in _segments(_db(wav[:, c], hop) > VAD_DB)]
    segs.sort()
    blocks = []
    for s, e, c in segs:
        if blocks and s < blocks[-1][1]:
            b = blocks[-1]
            blocks[-1] = (b[0], max(b[1], e), b[2], c if e >= b[1] else b[3])
        else:
            blocks.append((s, e, c, c))
    # pad each block (soft onsets / whisper timestamps lead the energy), never into a neighbour
    out = []
    dur = len(wav) / sr
    for i, (s, e, c0, c1) in enumerate(blocks):
        lo = 0.0 if i == 0 else (blocks[i - 1][1] + s) / 2
        hi = dur if i == len(blocks) - 1 else (e + blocks[i + 1][0]) / 2
        out.append((max(lo, s - BLOCK_PAD_S), min(hi, e + BLOCK_PAD_S), c0, c1))
    return out


def frame_speech(wav: np.ndarray, sr: int) -> np.ndarray:
    """wav [N, 2] -> bool [2, F] at Mimi frame rate."""
    hop = int(sr / FRAME_RATE)
    return np.stack([_db(wav[:, c], hop) > FRAME_VAD_DB for c in range(2)])


def _layout(blocks, gaps):
    starts, t = [], LEAD_S
    for i, (s, e, *_) in enumerate(blocks):
        starts.append(t)
        t += (e - s) + (gaps[i] if i < len(gaps) else 0)
    return starts, t + TAIL_S


def rebuild(wav, sr, alignments, target: float | None):
    """Returns new wav [N,2], new alignments, measured both-silent ratio.

    Only gaps at speaker changes are rescaled (common factor k); pauses
    inside one speaker's turn keep their original length.
    """
    blocks = speech_blocks(wav, sr)
    orig_gaps = np.array([max(blocks[i + 1][0] - blocks[i][1], 0.0) for i in range(len(blocks) - 1)])
    cross = np.array([blocks[i][3] != blocks[i + 1][2] for i in range(len(blocks) - 1)], dtype=bool)

    def build(k):
        gaps = np.where(cross, np.maximum(orig_gaps * k, MIN_GAP_S), orig_gaps)
        starts, total = _layout(blocks, gaps)
        out = np.zeros((int(math.ceil(total * sr)) + 1, 2), dtype=np.float32)
        for (s, e, *_), ns in zip(blocks, starts):
            a, b = int(round(s * sr)), int(round(e * sr))
            o = int(round(ns * sr))
            out[o:o + (b - a)] = wav[a:b]
        return out, starts

    def ratio(x):
        fs = frame_speech(x, sr)
        return float((~(fs[0] | fs[1])).mean())

    if target is None:
        out, starts = build(1.0)
    else:
        lo, hi = 0.0, 1.0
        while ratio(build(hi)[0]) < target and hi < 1e3:
            hi *= 2
        for _ in range(25):
            mid = (lo + hi) / 2
            if ratio(build(mid)[0]) < target:
                lo = mid
            else:
                hi = mid
        out, starts = build(hi)

    new_al = []
    for word, (ws, we), spk in alignments:
        ov = [min(we, e) - max(ws, s) for s, e, *_ in blocks]
        bi = int(np.argmax(ov))
        if ov[bi] <= 0:   # no overlap: nearest block
            bi = int(np.argmin([min(abs(ws - e), abs(we - s)) for s, e, *_ in blocks]))
        shift = starts[bi] - blocks[bi][0]
        new_al.append([word, [ws + shift, we + shift], spk])
    return out, new_al, ratio(out)


# --- text stream: copy of moshi-finetune Interleaver (keep_main_only=True) -----
def text_stream(tokenizer, alignments, T, text_padding=3, end_of_text_padding=0,
                main_speaker="SPEAKER_MAIN"):
    al = []
    for word, ts, spk in sorted(alignments, key=lambda x: x[1][0]):
        if spk != main_speaker or not ts[0] < ts[1]:
            continue
        al.append((tokenizer.encode(word.strip()), ts))
    toks = [text_padding] * T
    i, stack, last_word_end = 0, deque(), -1
    for t in range(T):
        while i < len(al) and al[i][1][0] * FRAME_RATE < t + 1:
            last_word_end = int(al[i][1][1] * FRAME_RATE)
            stack = deque(al[i][0])
            i += 1
        if stack:
            if t > 0 and toks[t - 1] == text_padding:
                toks[t - 1] = end_of_text_padding
            toks[t] = stack.popleft()
        elif t <= last_word_end:
            toks[t] = text_padding
    return torch.tensor(toks, dtype=torch.long)


@torch.no_grad()
def encode_dialog(mimi, tokenizer, wav, sr, alignments, device):
    """wav [N,2] float -> codes [17, F] (text, moshi x8, user x8), speech [2, F]."""
    import torchaudio.functional as AF
    x = torch.from_numpy(wav.T.copy()).float()
    x = AF.resample(x, sr, SR_MIMI)
    hop = int(SR_MIMI / FRAME_RATE)
    n = int(math.ceil(x.shape[1] / hop)) * hop
    x = torch.nn.functional.pad(x, (0, n - x.shape[1]))
    audio = mimi.encode(x[:, None].to(device))          # [2, 8, F]
    F_ = audio.shape[-1]
    text = text_stream(tokenizer, alignments, F_)
    codes = torch.cat([text[None].to(device), audio.reshape(16, F_)], 0)
    speech = torch.from_numpy(frame_speech(x.T.numpy(), SR_MIMI)[:, :F_].copy())
    if speech.shape[1] < F_:
        speech = torch.nn.functional.pad(speech, (0, F_ - speech.shape[1]))
    return codes.cpu().to(torch.int16), speech.bool()


def load_dialog(root, rel_wav):
    import soundfile as sf
    wav, sr = sf.read(f"{root}/{rel_wav}", dtype="float32")
    with open(f"{root}/{rel_wav[:-4]}.json") as f:
        al = json.load(f)["alignments"]
    return wav, sr, al
