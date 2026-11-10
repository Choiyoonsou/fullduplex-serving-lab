"""Export a few dialogs + results for the demo page (demo/).

  python scripts/05_export_demo.py --ids 1169 146 2067 98
Writes demo/data.json and demo/audio/{id}_r{ratio}.mp3 (rebuilt stereo dialog:
left = Moshi channel, right = user channel).
"""
import argparse
import base64
import glob
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from fdkv.data import load_dialog, rebuild  # noqa: E402
from fdkv.policies import make_policies  # noqa: E402
from fdkv.results import load_summary  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ids", nargs="+", default=None)
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--res", default="runs/yschoi/20261110-0510_baseline/nll")
ap.add_argument("--out", default="demo")
ap.add_argument("--root", default="data/DailyTalkContiguous")
ap.add_argument("--test_res", default=None, help="test-split run nll dir, for the findings table (dev = --res)")
ap.add_argument("--whisper", default="small.en", help="faster-whisper model for the user-channel transcript")
args = ap.parse_args()

os.makedirs(f"{args.out}/audio", exist_ok=True)
summ = load_summary(args.res)
ids = args.ids
if not ids:   # dialogs near the median length, so all three ratios are typical
    L = summ[summ.policy == "full_s3000"].groupby("id").frames.mean().sort_values()
    mid = len(L) // 2
    ids = [str(i) for i in L.index[mid - args.n // 2: mid + (args.n + 1) // 2]]

import sentencepiece as spm  # noqa: E402
tok = spm.SentencePieceProcessor(glob.glob("hf_cache/hub/models--kyutai--moshiko-pytorch-bf16/snapshots/*/tokenizer_spm_32k_3.model")[0])
pols = [p for p in make_policies() if p.kind != "full" or p.kv_bits is None]


def b64u8(x, scale):
    x = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0)
    return base64.b64encode(np.clip(np.round(x * scale), 0, 255).astype(np.uint8).tobytes()).decode()


def b64bits(b):
    return base64.b64encode(np.asarray(b, dtype=np.uint8).tobytes()).decode()


_whisper = None


def user_words(w2, sr, cache_fp):
    """Whisper transcript of the user channel (ch1) with word times in seconds.

    DailyTalkContiguous only aligns the main (Moshi) speaker, so the demo's chat
    view gets the other side from ASR. Display only; never used in experiments.
    """
    global _whisper
    if os.path.exists(cache_fp):
        return json.load(open(cache_fp))
    import torchaudio.functional as AF
    from faster_whisper import WhisperModel
    if _whisper is None:
        _whisper = WhisperModel(args.whisper, device="cpu", compute_type="int8", cpu_threads=16)
    x = AF.resample(torch.from_numpy(np.ascontiguousarray(w2[:, 1])).float(), sr, 16000).numpy()
    segs, _ = _whisper.transcribe(x, language="en", word_timestamps=True, vad_filter=True,
                                  condition_on_previous_text=False, beam_size=5)
    out = [[round(w.start, 2), w.word] for sg in segs for w in (sg.words or [])]
    os.makedirs(os.path.dirname(cache_fp), exist_ok=True)
    json.dump(out, open(cache_fp, "w"))
    return out


def build_turns(w2, sr, moshi_al, uwords, merge_s=1.0):
    """Chat turns from each channel's own speech segments (VAD), not from word gaps.

    Word timestamps can be off (the dataset's first Moshi word often starts far too
    early), so each word is snapped into the nearest speech segment of its own
    channel; a turn = one segment (pauses < merge_s merged). -> [{spk, words: [[frame, text]]}]
    """
    from fdkv.data import VAD_DB, VAD_HOP_S, _db
    hop = int(sr * VAD_HOP_S)
    turns = []
    for ch, spk, words in ((0, "M", [(w[1][0], w[1][1], w[0]) for w in moshi_al if w[2] == "SPEAKER_MAIN"]),
                           (1, "U", [(t, t, txt) for t, txt in uwords])):
        sp = _db(w2[:, ch], hop) > VAD_DB
        idx = np.flatnonzero(sp)
        segs = []
        for i in idx:
            t = i * VAD_HOP_S
            if segs and t - segs[-1][1] <= merge_s:
                segs[-1][1] = t + VAD_HOP_S
            else:
                segs.append([t, t + VAD_HOP_S])
        buckets = [[] for _ in segs]
        for ws, we, txt in words:
            if not segs:
                break
            ov = [min(we, e) - max(ws, s0) for s0, e in segs]
            k = int(np.argmax(ov)) if max(ov) > 0 else int(np.argmin([min(abs(ws - e), abs(ws - s0)) for s0, e in segs]))
            t = min(max(ws, segs[k][0]), segs[k][1])
            buckets[k].append((t, txt.strip()))
        for (s0, _), bw in zip(segs, buckets):
            if bw:
                bw.sort()
                turns.append({"spk": spk, "t0": s0, "words": [[int(t * 12.5), txt] for t, txt in bw]})
    turns.sort(key=lambda x: x["t0"])
    for tr in turns:
        tr.pop("t0")
    return turns


dialogs = []
with open(f"{args.root}/dailytalk.jsonl") as f:
    paths = {os.path.splitext(os.path.basename(json.loads(l)["path"]))[0]: json.loads(l)["path"] for l in f}
for did in ids:
    wav, sr, al = load_dialog(args.root, paths[did])
    entry = {"id": did, "ratios": {}}
    for r in (0.2, 0.4, 0.6):
        rt = f"r{r:.1f}"
        d = torch.load(f"data/prepared/{rt}/{did}.pt")
        z = np.load(f"{args.res}/{rt}_{did}.npz")
        S = d["codes"].shape[1]
        mp3 = f"audio/{did}_{rt}.mp3"
        w2, al2, _ = rebuild(wav, sr, al, r)   # deterministic: same audio as the experiment
        turns = build_turns(w2, sr, al2, user_words(w2, sr, f"data/demo_cache/{did}_{rt}_user.json"))
        if not os.path.exists(f"{args.out}/{mp3}"):
            tmp = f"{args.out}/audio/tmp.wav"
            sf.write(tmp, w2, sr)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-ac", "2", "-ar", "24000",
                            "-b:a", "48k", f"{args.out}/{mp3}"], check=True)
            os.remove(tmp)
        tt = d["codes"][0].long().numpy()
        words = [[int(t), tok.id_to_piece(int(x)).replace("▁", " ")] for t, x in enumerate(tt) if x > 3]
        nll = {}
        for p in pols:
            nll[p.tag] = {"text": b64u8(z[f"{p.tag}/text"], 20), "cb0": b64u8(z[f"{p.tag}/cb0"], 20)}
        entry["ratios"][rt] = {"S": int(S), "ratio": round(float(d["ratio"]), 3), "audio": mp3,
                               "speech0": b64bits(d["speech"][0].numpy()),
                               "speech1": b64bits(d["speech"][1].numpy()),
                               "keep": b64bits(z["keep"]), "words": words, "turns": turns, "nll": nll}
    dialogs.append(entry)
    print("exported", did, flush=True)

tab = pd.read_csv(f"{args.res}/analysis/table_all.csv")
tab = tab[["ratio", "policy", "kind", "budget", "kv_mb", "sessions_3090", "d_text_nll_content",
           "d_text_nll_content_lo", "d_text_nll_content_hi", "d_cb0_nll", "d_cb0_nll_lo", "d_cb0_nll_hi", "n"]]
policies = [{"tag": p.tag, "kind": p.kind, "slots": p.slots, "n_sink": p.n_sink, "skip": p.skip_silence,
             "bits": p.kv_bits, "recent": p.recent, "kv_mb": round(p.kv_mb(), 2),
             "budget": int(round(p.bf16_equiv_frames())) if p.kind != "full" else 3000} for p in pols]

def hyp(res):
    if not res or not os.path.exists(f"{res}/analysis/hypotheses.csv"):
        return None
    h = pd.read_csv(f"{res}/analysis/hypotheses.csv")
    h = h[["family", "contrast", "metric", "ratio", "budget", "n", "mean_diff", "ci95_lo", "ci95_hi", "dz", "p_holm", "significant"]]
    e = pd.read_csv(f"{res}/analysis/hypotheses_h4a_equivalence.csv")
    return {"tests": h.round(4).to_dict("records"), "equivalence": e.round(5).to_dict("records"),
            "n": int(load_summary(res).id.nunique())}


json.dump({"dialogs": dialogs, "policies": policies, "table": tab.round(4).to_dict("records"),
           "n_dialogs": int(summ.id.nunique()),
           "hypotheses": {"dev": hyp(args.res), "test": hyp(args.test_res)}}, open(f"{args.out}/data.json", "w"), separators=(",", ":"))
print("size MB", os.path.getsize(f"{args.out}/data.json") / 1e6)
