"""Load a run's per-dialog summary rows.

During a run, each worker appends to nll/summary_shard{i}.jsonl. When the run
finishes, run_all.sh merges them into nll/summary.jsonl.gz (8x smaller); only
the .gz is committed to git. Readers accept either form.
"""
import glob
import os

import pandas as pd


def load_summary(res: str) -> pd.DataFrame:
    parts = []
    gz = os.path.join(res, "summary.jsonl.gz")
    if os.path.exists(gz):
        parts.append(pd.read_json(gz, lines=True, compression="gzip"))
    parts += [pd.read_json(f, lines=True) for f in sorted(glob.glob(os.path.join(res, "summary_shard*.jsonl")))]
    if not parts:
        raise SystemExit(f"no summary in {res} (expected summary.jsonl.gz or summary_shard*.jsonl)")
    d = pd.concat(parts, ignore_index=True)
    d["id"] = d["id"].astype(str)
    d = d.drop_duplicates(["id", "ratio_target", "policy"], keep="last")
    # fixed row order: bootstrap/permutation results must not depend on file or shard order
    d["_idnum"] = pd.to_numeric(d["id"], errors="coerce")
    return d.sort_values(["ratio_target", "policy", "_idnum", "id"]).drop(columns="_idnum").reset_index(drop=True)


def merge_shards(res: str) -> str:
    """Merge summary_shard*.jsonl (+ existing .gz) into summary.jsonl.gz, then delete the shards."""
    d = load_summary(res)
    out = os.path.join(res, "summary.jsonl.gz")
    tmp = out + ".tmp"
    d.to_json(tmp, orient="records", lines=True, force_ascii=False, compression="gzip")
    os.replace(tmp, out)
    for f in glob.glob(os.path.join(res, "summary_shard*.jsonl")):
        os.remove(f)
    return out


if __name__ == "__main__":
    import sys
    print(merge_shards(sys.argv[1]))
