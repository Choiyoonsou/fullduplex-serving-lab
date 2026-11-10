"""Phase 1 main experiment: teacher-forced NLL for every (dialog, silence ratio, policy).

One worker = one GPU pair (T4) or one GPU twice (3090). Shard across workers:
  python scripts/03_run_nll.py --gpus 0 1 --shard 0 --nshards 3   (see scripts/run_nll.sh)

Efficiency:
  * each policy = one parallel forward with a policy mask (exact, see 02_verify)
  * policies whose mask equals an already-computed one (e.g. budget >= dialog
    length, or P1 when nothing is silent) are aliased, not recomputed
  * remaining policies for a dialog are batched (--batch) into one forward
  * per-frame NLLs are saved (npz) so any re-slicing of results is offline
  * resumable: dialogs with an existing npz are skipped
"""
import argparse
import glob
import json
import os
import sys
import time

os.environ.setdefault("HF_HOME", os.path.abspath("hf_cache"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from fdkv.model import CTX, load_moshi, per_frame_nll, teacher_forced  # noqa: E402
from fdkv.policies import KV_KINDS, attention_mask, keep_from_speech, make_policies, variant_policies  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/prepared")
ap.add_argument("--out", required=True, help="e.g. runs/<member>/<run>/nll (scripts/run_all.sh sets this)")
ap.add_argument("--gpus", type=int, nargs=2, default=[0, 1])
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--batch", type=int, default=3)
ap.add_argument("--budgets", type=int, nargs="+", default=[125, 250, 500, 1000])
ap.add_argument("--n_sink", type=int, default=4)
ap.add_argument("--hang", type=int, default=1, help="speech dilation (frames) for P1 keep mask")
ap.add_argument("--int4", action="store_true")
ap.add_argument("--limit", type=int, default=0, help="debug: only first N files of this shard")
ap.add_argument("--ids_file", default=None, help="only these dialog ids (data/prepared is shared by all runs)")
ap.add_argument("--ratios", type=float, nargs="+", default=None, help="only these silence ratios")
ap.add_argument("--exp", default="kv_policy", choices=["kv_policy", "rvq", "kv_quant"],
                help="kv_policy: KV policy grid | rvq: Moshi codebooks k..7 removed from the input | kv_quant: KV bit-width sweep")
ap.add_argument("--kinds", nargs="+", default=list(KV_KINDS), help="kv_policy: which policy kinds")
ap.add_argument("--rvq_ks", type=int, nargs="+", default=[6, 4, 2, 1], help="rvq: codebook counts to keep")
ap.add_argument("--kv_bits", type=int, nargs="+", default=[8, 6, 4, 3], help="kv_quant: KV bit widths")
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
files = sorted(glob.glob(f"{args.data}/r*/*.pt"))
if args.ids_file:
    keep_ids = set(open(args.ids_file).read().split())
    files = [f for f in files if os.path.splitext(os.path.basename(f))[0] in keep_ids]
if args.ratios:
    files = [f for f in files if any(f"/r{r:.1f}/" in f for r in args.ratios)]
files = files[args.shard::args.nshards]
if args.limit:
    files = files[: args.limit]
if args.exp == "kv_policy":
    policies = make_policies(tuple(args.budgets), args.n_sink, args.int4, tuple(args.kinds))
elif args.exp == "rvq":
    policies = variant_policies(rvq_ks=args.rvq_ks)
else:
    policies = variant_policies(kv_bits=args.kv_bits)
print(f"shard {args.shard}/{args.nshards}: {len(files)} files x {len(policies)} policies")

lm, _ = load_moshi(f"cuda:{args.gpus[0]}", f"cuda:{args.gpus[1]}", torch.float16)
summary_fp = f"{args.out}/summary_shard{args.shard}.jsonl"
CONTENT_EXCLUDE = {0, 3}   # EPAD, PAD


TURN_GAP = 12              # frames (~1s) without Moshi text before a turn onset


def turn_masks(tgt_text, speech):
    """Frames that carry turn-taking decisions in Moshi's text stream.
    onset : first content token of a Moshi turn (no content in the previous ~1s)
    epad  : EPAD target = "start a word at the next frame" decision
    listen: PAD target while only the user is speaking = "keep quiet, don't barge in"
    """
    content = ~np.isin(tgt_text, list(CONTENT_EXCLUDE))
    prev = np.convolve(content.astype(int), np.ones(TURN_GAP, int), "full")[:len(content)] - content
    onset = content & (prev == 0)
    sp = speech.numpy().astype(bool)
    listen = (tgt_text == 3) & sp[1] & ~sp[0]
    return {"content": content, "onset": onset, "epad": tgt_text == 0, "listen": listen}


def summarize(n, masks):
    text = n["text"]
    sel = lambda m: float(np.nanmean(np.where(m, text, np.nan))) if m.any() else None  # noqa: E731
    return {
        "text_nll": float(np.nanmean(text)),
        "text_nll_content": sel(masks["content"]),
        "text_nll_onset": sel(masks["onset"]),
        "text_nll_epad": sel(masks["epad"]),
        "text_nll_listen": sel(masks["listen"]),
        "cb0_nll": float(np.nanmean(n["audio"][0])),
        "audio_nll": float(np.nanmean(n["audio"])),
    }


for fi, fp in enumerate(files):
    d = torch.load(fp)
    rtag = os.path.basename(os.path.dirname(fp))
    out_npz = f"{args.out}/{rtag}_{d['id']}.npz"
    if os.path.exists(out_npz):
        continue
    t0 = time.time()
    codes = d["codes"][None].long()
    S = codes.shape[-1]
    keep = keep_from_speech(d["speech"], args.hang)
    tgt_text = codes[0, 0].numpy()
    masks = turn_masks(tgt_text, d["speech"])

    # dedupe policies by (mask, bits, rvq_k)
    uniq, alias = [], {}
    for p in policies:
        m = attention_mask(p, keep)
        for ui, (um, ub, uk) in enumerate(uniq):
            if ub == p.kv_bits and uk == p.rvq_k and torch.equal(um, m):
                alias[p.tag] = ui
                break
        else:
            alias[p.tag] = len(uniq)
            uniq.append((m, p.kv_bits, p.rvq_k))

    results = [None] * len(uniq)
    for s in range(0, len(uniq), args.batch):
        chunk = uniq[s:s + args.batch]
        CTX.set(torch.stack([m for m, _, _ in chunk]), [b for _, b, _ in chunk])
        cb = codes.expand(len(chunk), -1, -1)
        cin = None
        if any(k < 8 for _, _, k in chunk):   # RVQ: Moshi codebooks k..7 absent from the input
            cin = cb.clone()
            for j, (_, _, k) in enumerate(chunk):
                cin[j, 1 + k:9] = lm.zero_token_id
        n = per_frame_nll(teacher_forced(lm, cb, codes_in=cin), cb)
        CTX.clear()
        for j in range(len(chunk)):
            results[s + j] = {"text": n["text"][j], "audio": n["audio"][j]}

    arrays = {"keep": keep.numpy(), "target_text": tgt_text}
    with open(summary_fp, "a") as f:
        for p in policies:
            ui = alias[p.tag]
            r = results[ui]
            m = uniq[ui][0]
            arrays[f"{p.tag}/text"] = r["text"].astype(np.float16)
            arrays[f"{p.tag}/cb0"] = r["audio"][0].astype(np.float16)
            arrays[f"{p.tag}/audio"] = np.nanmean(r["audio"], 0).astype(np.float16)
            row = {"id": d["id"], "exp": args.exp, "ratio_target": d["ratio_target"], "ratio": d["ratio"], "frames": S,
                   "keep_frac": float(keep.float().mean()), "policy": p.tag, "kind": p.kind,
                   "slots": p.slots, "n_sink": p.n_sink, "bits": p.kv_bits, "rvq_k": p.rvq_k, "kv_mb": p.kv_mb(),
                   "mem_bf16_frames": p.bf16_equiv_frames(),
                   "visible_mean": float(m.sum(1).float().mean()), "computed_as": ui}
            row.update(summarize(r, masks))
            f.write(json.dumps(row) + "\n")
    np.savez_compressed(out_npz, **arrays)
    print(f"[{fi + 1}/{len(files)}] {rtag}/{d['id']} S={S} unique={len(uniq)}/{len(policies)} "
          f"{time.time() - t0:.1f}s", flush=True)
