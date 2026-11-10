"""Download moshiko weights and the N longest DailyTalkContiguous dialogs.

  python scripts/00_download.py                    # model + dialogs in splits/dev.txt, splits/test.txt
  python scripts/00_download.py --skip_model

Anonymous Hugging Face downloads get rate limited (HTTP 429); this retries
with backoff. Set HF_TOKEN to avoid it.
"""
import argparse
import json
import os
import time

os.environ.setdefault("HF_HOME", os.path.abspath("hf_cache"))
from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402

MODEL_REPO = "kyutai/moshiko-pytorch-bf16"
DATA_REPO = "kyutai/DailyTalkContiguous"

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=None, help="download the N longest dialogs instead of splits/*.txt (0 = all 2541)")
ap.add_argument("--root", default="data/DailyTalkContiguous")
ap.add_argument("--skip_model", action="store_true")
ap.add_argument("--retries", type=int, default=30)
args = ap.parse_args()


def retry(fn, what):
    for i in range(args.retries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001  (429s and transient network errors)
            wait = min(300, 30 * (i + 1))
            print(f"[{what}] {type(e).__name__}: {str(e)[:120]} -> retry in {wait}s ({i + 1}/{args.retries})")
            time.sleep(wait)
    raise SystemExit(f"[{what}] failed after {args.retries} retries")


if not args.skip_model:
    print(f"model: {MODEL_REPO} -> {os.environ['HF_HOME']} (~16GB)")
    retry(lambda: snapshot_download(MODEL_REPO), "model")

os.makedirs(args.root, exist_ok=True)
jl = retry(lambda: hf_hub_download(DATA_REPO, "dailytalk.jsonl", repo_type="dataset", local_dir=args.root), "index")
items = sorted((json.loads(l) for l in open(jl)), key=lambda x: -x["duration"])
if args.n is None:   # default: exactly the dev + test splits
    import glob
    want = set()
    for f in glob.glob("splits/*.txt"):
        want |= set(open(f).read().split())
    items = [it for it in items if os.path.splitext(os.path.basename(it["path"]))[0] in want]
elif args.n:
    items = items[: args.n]
patterns = [it["path"][:-4] + ext for it in items for ext in (".wav", ".json")]
missing = [p for p in patterns if not os.path.exists(os.path.join(args.root, p))]
print(f"data: {len(items)} dialogs ({items[-1]['duration']:.0f}s..{items[0]['duration']:.0f}s), {len(missing)} files to fetch")
while missing:
    retry(lambda: snapshot_download(DATA_REPO, repo_type="dataset", local_dir=args.root,
                                    allow_patterns=missing[:200]), "data")
    missing = [p for p in patterns if not os.path.exists(os.path.join(args.root, p))]
print("download complete")
