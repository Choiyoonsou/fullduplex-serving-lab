"""Phase 1 data: pick N dialogs, rebuild at each silence ratio, Mimi-encode once.

  python scripts/01_prep_data.py --n 100 --ratios 0.2 0.4 0.6 --gpu 0
Output: data/prepared/r{ratio}/{dialog_id}.pt with codes [17,F] int16,
speech [2,F] bool, measured ratio. Encoded once, reused by every policy.
"""
import argparse
import json
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fdkv.data import encode_dialog, load_dialog, rebuild  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/DailyTalkContiguous")
ap.add_argument("--out", default="data/prepared")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--ratios", type=float, nargs="+", default=[0.2, 0.4, 0.6])
ap.add_argument("--max_frames", type=int, default=3000, help="clip to moshi context")
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--ids_file", default=None, help="use exactly these dialog ids (e.g. splits/test.txt); --n then takes the first N")
ap.add_argument("--ids_out", default=None, help="write the selected dialog ids here (one per line)")
args = ap.parse_args()

os.environ.setdefault("HF_HOME", os.path.abspath("hf_cache"))
from moshi.models import loaders  # noqa: E402

dev = f"cuda:{args.gpu}"
ci = loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
mimi = ci.get_mimi(device=dev)
tok = ci.get_text_tokenizer()

items = [json.loads(l) for l in open(f"{args.root}/dailytalk.jsonl")]
items = [it for it in items if os.path.exists(f"{args.root}/{it['path']}")]
items.sort(key=lambda it: -it["duration"])   # longest first: budgets must bind
if args.ids_file:
    order = open(args.ids_file).read().split()
    by_id = {os.path.splitext(os.path.basename(it["path"]))[0]: it for it in items}
    missing = [i for i in order if i not in by_id]
    assert not missing, f"{len(missing)} dialogs not downloaded (run scripts/00_download.py): {missing[:5]}"
    items = [by_id[i] for i in order]
items = items[: args.n]
print(f"{len(items)} dialogs, {items[-1]['duration']:.1f}s..{items[0]['duration']:.1f}s")
if args.ids_out:
    os.makedirs(os.path.dirname(os.path.abspath(args.ids_out)), exist_ok=True)
    with open(args.ids_out, "w") as f:
        f.write("\n".join(os.path.splitext(os.path.basename(it["path"]))[0] for it in items) + "\n")

meta = []
for it in tqdm(items):
    did = os.path.splitext(os.path.basename(it["path"]))[0]
    wav, sr, al = load_dialog(args.root, it["path"])
    for r in args.ratios:
        od = f"{args.out}/r{r:.1f}"
        os.makedirs(od, exist_ok=True)
        fp = f"{od}/{did}.pt"
        if os.path.exists(fp):
            continue
        w2, al2, measured = rebuild(wav, sr, al, r)
        codes, speech = encode_dialog(mimi, tok, w2, sr, al2, dev)
        codes, speech = codes[:, : args.max_frames], speech[:, : args.max_frames]
        tmp = f"{fp}.{os.getpid()}.tmp"   # atomic: data/prepared is shared by concurrent runs
        torch.save({"codes": codes, "speech": speech, "ratio_target": r, "ratio": measured,
                    "id": did, "dur_s": len(w2) / sr}, tmp)
        os.replace(tmp, fp)
        meta.append({"id": did, "target": r, "ratio": measured, "frames": codes.shape[1]})

with open(f"{args.out}/meta.jsonl", "a") as f:
    for m in meta:
        f.write(json.dumps(m) + "\n")
if meta:
    import pandas as pd
    print(pd.DataFrame(meta).groupby("target")[["ratio", "frames"]].describe().T)
