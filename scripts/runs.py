"""List and compare experiment runs (runs/<member>/<run>/).

  python scripts/runs.py                       # all runs: who, when, status, settings
  python scripts/runs.py --member jmkim        # one person's runs
  python scripts/runs.py compare               # ΔNLL side by side, latest finished run of each member
  python scripts/runs.py compare runs/jmkim/20261111-0930_hang2 runs/yschoi/20261110-0510_baseline \\
         --ratio 0.6 --kinds B1 P1W P2
"""
import argparse
import glob
import json
import os

import pandas as pd
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from fdkv.results import load_summary  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("cmd", nargs="?", default="list", choices=["list", "compare"])
ap.add_argument("runs", nargs="*", help="run dirs for compare (default: latest done run per member)")
ap.add_argument("--member", default=None)
ap.add_argument("--ratio", type=float, default=0.6)
ap.add_argument("--kinds", nargs="+", default=["B1", "P1S", "P1W", "B2", "P2"])
ap.add_argument("--metric", default="d_text_nll_content", choices=["d_text_nll_content", "d_cb0_nll"])
args = ap.parse_args()

members = json.load(open("members.json"))["members"]
rows = []
for mp in sorted(glob.glob("runs/*/*/meta.json")):
    m = json.load(open(mp))
    d = os.path.dirname(mp)
    try:
        done = load_summary(f"{d}/nll").groupby(["id", "ratio_target"]).ngroups
    except SystemExit:
        done = 0
    rows.append({"run": d, "who": m.get("name"), "member": m["member"], "tag": m["tag"], "split": m.get("split", "dev"),
                 "started": m["started"], "status": m["status"], "commit": m.get("git_commit"),
                 "dirty": m.get("git_dirty"), "dialogs": m["n_dialogs"],
                 "args": " ".join(m.get("nll_args", [])), "note": m.get("note", ""), "items_done": done})
runs = pd.DataFrame(rows)
if runs.empty:
    raise SystemExit("no runs yet. start one: MEMBER=<id> TAG=<name> scripts/run_all.sh")
if args.member:
    runs = runs[runs.member == args.member]
pd.set_option("display.width", 220, "display.max_colwidth", 60)

if args.cmd == "list":
    print(runs.drop(columns=["member"]).to_string(index=False))
    raise SystemExit

sel = args.runs or [g.sort_values("started").run.iloc[-1] for _, g in runs[runs.status == "done"].groupby("member")]
cols = {}
for r in sel:
    p = f"{r}/nll/analysis/table_all.csv"
    if not os.path.exists(p):
        print(f"skip {r}: no analysis (run scripts/04_analyze.py --res {r}/nll)")
        continue
    t = pd.read_csv(p)
    t = t[(abs(t.ratio - args.ratio) < 1e-6) & t.kind.isin(args.kinds)]
    m = json.load(open(f"{r}/meta.json"))
    label = f"{m['name']}:{m['tag']}"
    cols[label] = t.set_index(["kind", "budget"])[args.metric]
if not cols:
    raise SystemExit("nothing to compare")
out = pd.DataFrame(cols).sort_index()
print(f"{args.metric} (vs full context), silence {int(args.ratio * 100)}%  [lower is better]")
print(out.round(3).to_string())
