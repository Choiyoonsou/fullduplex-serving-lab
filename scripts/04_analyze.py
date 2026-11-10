"""Aggregate NLL results -> tables + the main figure (memory budget vs ΔNLL).

  python scripts/04_analyze.py --res runs/<member>/<run>/nll --kv_budget_gb 6.0

ΔNLL is paired per dialog: policy NLL - full-context NLL of the same dialog.
95% CI = bootstrap over dialogs. Sessions = floor(KV budget / per-session KV),
using the Phase 0 3090 measurement (~6GB left for KV after weights; 0.5MB/frame bf16).
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from fdkv.results import load_summary  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--res", default="runs/yschoi/20261110-0510_baseline/nll")
ap.add_argument("--kv_budget_gb", type=float, default=6.0)
ap.add_argument("--tau", type=float, default=0.05, help="max acceptable ΔNLL(content) for the capacity table")
args = ap.parse_args()

d = load_summary(args.res)
METRICS = ["text_nll_content", "cb0_nll", "text_nll", "audio_nll"]
full = d[(d.kind == "full") & d.bits.isna()].set_index(["id", "ratio_target"])[METRICS]
for m in METRICS:
    d["d_" + m] = d[m].values - full.loc[list(zip(d.id, d.ratio_target)), m].values
d["budget"] = d.mem_bf16_frames.round().astype(int)
d.loc[d.kind == "full", "budget"] = 3000


def boot(x, n=2000, seed=0):
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, (n, len(x))).mean(1)
    return pd.Series({"mean": x.mean(), "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5), "n": len(x)})


rows = []
for (r, pol), g in d.groupby(["ratio_target", "policy"]):
    rec = {"ratio": r, "policy": pol, "kind": g.kind.iloc[0], "budget": g.budget.iloc[0],
           "kv_mb": g.kv_mb.iloc[0], "visible": g.visible_mean.mean(), "keep_frac": g.keep_frac.mean()}
    for m in ["d_text_nll_content", "d_cb0_nll"]:
        b = boot(g[m])
        rec[m], rec[m + "_lo"], rec[m + "_hi"] = b["mean"], b["lo"], b["hi"]
    rec["n"] = len(g)
    rows.append(rec)
t = pd.DataFrame(rows).sort_values(["ratio", "budget", "kind"])
t["sessions_3090"] = np.floor(args.kv_budget_gb * 1024 / t.kv_mb).astype(int)
os.makedirs(f"{args.res}/analysis", exist_ok=True)
t.to_csv(f"{args.res}/analysis/table_all.csv", index=False)
pd.set_option("display.width", 200)
print(t[["ratio", "kind", "budget", "kv_mb", "sessions_3090", "keep_frac", "d_text_nll_content",
         "d_text_nll_content_lo", "d_text_nll_content_hi", "d_cb0_nll", "n"]].round(3).to_string(index=False))

# capacity: max sessions per policy kind with both ΔNLLs (upper CI) <= tau
cap = []
for (r, k), g in t[t.kind != "full"].groupby(["ratio", "kind"]):
    ok = g[(g.d_text_nll_content_hi <= args.tau) & (g.d_cb0_nll_hi <= args.tau)]
    cap.append({"ratio": r, "kind": k, "max_sessions": int(ok.sessions_3090.max()) if len(ok) else 0,
                "at_budget": int(ok.loc[ok.sessions_3090.idxmax(), "budget"]) if len(ok) else None})
cap = pd.DataFrame(cap).pivot(index="kind", columns="ratio", values="max_sessions")
full_sessions = int(np.floor(args.kv_budget_gb * 1024 / t[t.policy == "full_s3000"].kv_mb.iloc[0]))
print(f"\nMax concurrent sessions on 3090 (KV {args.kv_budget_gb}GB) with ΔNLL upper-CI <= {args.tau}"
      f" (text-content AND cb0); full context = {full_sessions}")
print(cap.to_string())
cap.to_csv(f"{args.res}/analysis/capacity.csv")

# ---- figure: budget vs ΔNLL, one column per silence ratio, rows = metric ----
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"B0": "#2a78d6", "B1": "#eb6834", "B2": "#1baf7a", "P1": "#eda100",
          "P1S": "#e87ba4", "P1W": "#6250d6", "P2": "#e34948"}
LABEL = {"B0": "B0 ring", "B1": "B1 sink+window", "B2": "B2 B1+int8", "P1": "P1 skip-silence",
         "P1S": "P1S skip+sink", "P1W": "P1W skip+sink+recent", "P2": "P2 P1W+int8"}
ratios = sorted(t.ratio.unique())
fig, axes = plt.subplots(2, len(ratios), figsize=(4.2 * len(ratios), 7), sharex=True, squeeze=False)
for col, r in enumerate(ratios):
    for row, (m, name) in enumerate([("d_text_nll_content", "Δ text NLL (content tokens)"),
                                     ("d_cb0_nll", "Δ semantic cb0 NLL")]):
        ax = axes[row, col]
        for k, c in COLORS.items():
            g = t[(t.ratio == r) & (t.kind == k)].sort_values("budget")
            if g.empty:
                continue
            ax.plot(g.budget, g[m], color=c, lw=2, marker="o", ms=5, label=LABEL[k])
            ax.fill_between(g.budget, g[m + "_lo"], g[m + "_hi"], color=c, alpha=0.12, lw=0)
        ax.axhline(0, color="#888", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks([125, 250, 500, 1000])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.grid(alpha=0.25, lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if row == 0:
            ax.set_title(f"silence {int(r * 100)}%")
        if row == 1:
            ax.set_xlabel("KV memory per session (bf16 frames; 125 = 62MB)")
        if col == 0:
            ax.set_ylabel(name)
        ax.set_ylim(bottom=min(-0.05, ax.get_ylim()[0]), top=min(ax.get_ylim()[1], 0.6))
axes[0, -1].legend(fontsize=8, frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(f"{args.res}/analysis/fig1_budget_vs_dnll.png", dpi=150)
print(f"\nsaved {args.res}/analysis/ (table_all.csv, capacity.csv, fig1_budget_vs_dnll.png)")
