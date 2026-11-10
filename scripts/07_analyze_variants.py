"""Generic analysis: every policy/variant of a run vs the full-context reference.

  python scripts/07_analyze_variants.py --res runs/<member>/<run>/nll

Works for all experiment types (kv_policy, rvq, kv_quant). Per dialog paired
differences, bootstrap 95% CI, sign-flip permutation p with Holm correction
within metric x silence ratio. Includes the turn-taking metrics
(onset / EPAD / listen). Writes analysis/variants.csv, variants.md, fig_variants.png.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fdkv.results import load_summary  # noqa: E402

METRICS = [("text_nll_content", "텍스트(내용)"), ("cb0_nll", "음성 cb0"), ("text_nll_onset", "턴 시작"),
           ("text_nll_epad", "말 시작 결정(EPAD)"), ("text_nll_listen", "듣는 중 침묵 유지")]
EN = {"text_nll_content": "text (content)", "cb0_nll": "audio cb0", "text_nll_onset": "turn onset",
      "text_nll_epad": "EPAD (start-speaking)", "text_nll_listen": "stay quiet while listening"}
ap = argparse.ArgumentParser()
ap.add_argument("--res", required=True)
ap.add_argument("--ref", default="full_s3000")
ap.add_argument("--n_perm", type=int, default=10000)
args = ap.parse_args()
rng = np.random.default_rng(0)

d = load_summary(args.res)
d["ratio_target"] = d.ratio_target.round(2)
metrics = [(m, lab) for m, lab in METRICS if m in d.columns]
order = list(dict.fromkeys(d.policy))          # policy order as produced by the run
ref = d[d.policy == args.ref].set_index(["ratio_target", "id"])

rows = []
for (r, pol), g in d[d.policy != args.ref].groupby(["ratio_target", "policy"], sort=False):
    g = g.set_index(["ratio_target", "id"])
    for m, lab in metrics:
        x = (g[m] - ref.loc[g.index, m]).astype(float).dropna().to_numpy()
        if len(x) < 2:
            continue
        bs = x[rng.integers(0, len(x), (4000, len(x)))].mean(1)
        null = (rng.choice([-1.0, 1.0], (args.n_perm, len(x))) * np.abs(x)).mean(1)
        p = (np.sum(np.abs(null) >= abs(x.mean())) + 1) / (args.n_perm + 1)
        rows.append({"ratio": r, "policy": pol, "metric": m, "label": lab, "n": len(x), "mean_diff": x.mean(),
                     "ci95_lo": np.quantile(bs, .025), "ci95_hi": np.quantile(bs, .975), "p": p})
t = pd.DataFrame(rows)
t["p_holm"] = np.nan
for _, g in t.groupby(["metric", "ratio"]):
    idx = g.p.sort_values().index
    run = 0.0
    for i, j in enumerate(idx):
        run = max(run, min(1.0, (len(idx) - i) * t.at[j, "p"]))
        t.at[j, "p_holm"] = run
t["significant"] = t.p_holm < 0.05
t["order"] = t.policy.map({p: i for i, p in enumerate(order)})
t = t.sort_values(["ratio", "order", "metric"]).drop(columns="order")

out = f"{args.res}/analysis"
os.makedirs(out, exist_ok=True)
t.round(5).to_csv(f"{out}/variants.csv", index=False)

wide = t.assign(cell=lambda x: x.apply(lambda r: f"{r.mean_diff:+.3f}{'*' if r.significant else ''}", axis=1)) \
        .pivot_table(index=["ratio", "policy"], columns="label", values="cell", aggfunc="first", sort=False)
wide = wide[[lab for _, lab in metrics if lab in wide.columns]]
md = [f"# Variants vs `{args.ref}`: `{args.res}`", "",
      "ΔNLL (nats, variant − reference), dialog-paired; * = Holm-corrected p < 0.05 within metric × silence ratio.", "",
      "| silence | policy | " + " | ".join(wide.columns) + " |", "|---|---|" + "---|" * len(wide.columns)]
for (r, pol), row in wide.iterrows():
    md.append(f"| {r} | {pol} | " + " | ".join(str(v) for v in row.values) + " |")
open(f"{out}/variants.md", "w").write("\n".join(md) + "\n")
pd.set_option("display.width", 200)
print(wide.to_string())

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
pols = [p for p in order if p != args.ref]
if len(pols) <= 12:
    ratios = sorted(t.ratio.unique())
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.6 * len(metrics), 3.4), squeeze=False)
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
    for ax, (m, lab) in zip(axes[0], metrics):
        for ci, r in enumerate(ratios):
            g = t[(t.metric == m) & (t.ratio == r)].set_index("policy").reindex(pols)
            xs = np.arange(len(pols))
            ax.plot(xs, g.mean_diff, marker="o", lw=2, color=colors[ci % 4], label=f"silence {int(r * 100)}%")
            ax.fill_between(xs, g.ci95_lo, g.ci95_hi, color=colors[ci % 4], alpha=.12, lw=0)
        ax.axhline(0, color="#888", lw=1)
        ax.set_xticks(range(len(pols)), [p.replace("full_s3000_", "") for p in pols], rotation=30, fontsize=8)
        ax.set_title(EN.get(m, m), fontsize=10)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0][0].set_ylabel("ΔNLL vs reference")
    axes[0][-1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{out}/fig_variants.png", dpi=130)
print(f"saved {out}/variants.csv, variants.md")
