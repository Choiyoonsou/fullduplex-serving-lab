"""Pre-registered hypothesis tests (H1-H4) for one run.

  python scripts/06_hypotheses.py --res runs/<member>/<run>/nll

The comparisons, directions and families below are fixed in code BEFORE the
test split is evaluated; do not edit them after looking at test results.

Unit of analysis = dialog (frames within a dialog are not independent).
Per dialog: mean NLL over content text tokens / over all frames (cb0).
Test: paired sign-flip permutation test on the mean difference (N_PERM
resamples), one- or two-sided as specified. Holm-Bonferroni within each family.
Effect sizes: mean difference, 95% bootstrap CI, Cohen's dz, win rate.
H4a is an equivalence claim: 90% bootstrap CI inside ±EQ_MARGIN (TOST at α=.05).
H5 (real serving capacity) needs the 3090 serving measurement; not tested here.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from fdkv.results import load_summary  # noqa: E402

ALPHA = 0.05
N_PERM = 20000
N_BOOT = 5000
EQ_MARGIN = 0.01   # nats: "practically no change" for H4a
METRICS = {"text": "text_nll_content", "cb0": "cb0_nll"}
BUDGETS = [125, 250, 500, 1000]
BUDGETS_H2 = [125, 250, 500]

ap = argparse.ArgumentParser()
ap.add_argument("--res", required=True)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
rng = np.random.default_rng(args.seed)

d = load_summary(args.res)
d["ratio_target"] = d.ratio_target.round(2)
d["budget"] = (d.mem_bf16_frames / 125).round().astype(int) * 125
d.loc[(d.kind == "full") & d.bits.isna(), "kind"] = "full"
d.loc[(d.kind == "full") & d.bits.notna(), "kind"] = "full_int8"
ratios = sorted(d.ratio_target.unique())


def vec(kind, budget, ratio, metric):
    """Per-dialog metric, indexed by dialog id."""
    q = (d.kind == kind) & (d.ratio_target == ratio)
    if kind not in ("full", "full_int8"):
        q &= d.budget == budget
    return d[q].set_index("id")[METRICS[metric]].sort_index()


def perm_p(x, alternative):
    """Sign-flip permutation test of mean(x) against 0."""
    x = np.asarray(x)
    obs = x.mean()
    flips = rng.choice([-1.0, 1.0], size=(N_PERM, len(x)))
    null = (flips * np.abs(x)).mean(1)
    if alternative == "greater":
        return (np.sum(null >= obs) + 1) / (N_PERM + 1)
    if alternative == "less":
        return (np.sum(null <= obs) + 1) / (N_PERM + 1)
    return (np.sum(np.abs(null) >= abs(obs)) + 1) / (N_PERM + 1)


def boot_ci(x, level):
    x = np.asarray(x)
    bs = x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(1)
    a = (1 - level) / 2
    return np.quantile(bs, a), np.quantile(bs, 1 - a)


rows = []


def test(family, label, a, b, metric, ratio, budget, alternative, favors):
    """diff = a - b per dialog. `favors`: which sign means a is better (lower NLL)."""
    x = (a - b).dropna()
    lo, hi = boot_ci(x, 0.95)
    sd = x.std(ddof=1)
    rows.append({"family": family, "contrast": label, "metric": metric, "ratio": ratio, "budget": budget,
                 "n": len(x), "mean_diff": x.mean(), "ci95_lo": lo, "ci95_hi": hi,
                 "dz": x.mean() / sd if sd > 0 else np.nan,
                 "pct_neg": float((x < 0).mean()), "alternative": alternative,
                 "p": perm_p(x, alternative)})


for r in ratios:
    for m in METRICS:
        full = vec("full", None, r, m)
        # H1: shrinking memory (B1, the strong text-LLM baseline) raises NLL
        for c in BUDGETS:
            test("H1", f"B1@{c} - full", vec("B1", c, r, m), full, m, r, c, "greater", None)
        # H2: silence-aware KV beats B1 at equal memory (two-sided: cb0 may go either way)
        for c in BUDGETS_H2:
            test("H2", f"P1W@{c} - B1@{c}", vec("P1W", c, r, m), vec("B1", c, r, m), m, r, c, "two-sided", None)
            test("H2s", f"P1S@{c} - B1@{c}", vec("P1S", c, r, m), vec("B1", c, r, m), m, r, c, "two-sided", None)
        # H4b: combining with int8 helps at equal memory
        for c in BUDGETS_H2:
            test("H4b", f"P2@{c} - B2@{c}", vec("P2", c, r, m), vec("B2", c, r, m), m, r, c, "two-sided", None)
            test("H4b", f"P2@{c} - P1W@{c}", vec("P2", c, r, m), vec("P1W", c, r, m), m, r, c, "two-sided", None)

# H3: the P1W gain over B1 grows with silence ratio (same dialogs across ratios)
lo_r, hi_r = min(ratios), max(ratios)
if lo_r != hi_r:
    for m in METRICS:
        for c in BUDGETS_H2:
            g = lambda r: vec("B1", c, r, m) - vec("P1W", c, r, m)   # noqa: E731  gain (positive = P1W better)
            test("H3", f"gain@{hi_r:.1f} - gain@{lo_r:.1f}  (gain = B1@{c} - P1W@{c})",
                 g(hi_r), g(lo_r), m, f"{hi_r:.1f} vs {lo_r:.1f}", c, "greater", None)

res = pd.DataFrame(rows)
# Holm within family x metric
res["p_holm"] = np.nan
for _, g in res.groupby(["family", "metric"]):
    order = g.p.sort_values().index
    k = len(order)
    adj, running = [], 0.0
    for i, idx in enumerate(order):
        running = max(running, min(1.0, (k - i) * res.at[idx, "p"]))
        adj.append((idx, running))
    for idx, v in adj:
        res.at[idx, "p_holm"] = v
res["significant"] = res.p_holm < ALPHA

# H4a: equivalence of int8 KV with bf16 KV (full context)
eq = []
for r in ratios:
    for m in METRICS:
        x = (vec("full_int8", None, r, m) - vec("full", None, r, m)).dropna()
        lo, hi = boot_ci(x, 0.90)
        eq.append({"family": "H4a", "contrast": "full_int8 - full", "metric": m, "ratio": r, "n": len(x),
                   "mean_diff": x.mean(), "ci90_lo": lo, "ci90_hi": hi, "margin": EQ_MARGIN,
                   "equivalent": bool(lo > -EQ_MARGIN and hi < EQ_MARGIN)})
eq = pd.DataFrame(eq)

out = f"{args.res}/analysis"
os.makedirs(out, exist_ok=True)
res.to_csv(f"{out}/hypotheses.csv", index=False)
eq.to_csv(f"{out}/hypotheses_h4a_equivalence.csv", index=False)


def fmt(x):
    return f"{x:+.3f}"


lines = [f"# Hypothesis tests: `{args.res}`", "",
         f"Unit = dialog (n={res.n.max()}), paired sign-flip permutation ({N_PERM}), Holm within family×metric, α={ALPHA}.",
         "diff = first − second policy (negative = first has lower NLL = better).", ""]
verdict = {
    "H1": "memory reduction raises NLL (B1 vs full)",
    "H2": "P1W vs B1 at equal memory",
    "H2s": "P1S vs B1 at equal memory (secondary)",
    "H3": "P1W gain over B1 grows with silence ratio",
    "H4b": "int8 combination (P2 vs B2, P2 vs P1W)",
}
for fam, title in verdict.items():
    g = res[res.family == fam]
    if g.empty:
        continue
    lines += [f"## {fam}: {title}", "", "| contrast | metric | silence | mean diff [95% CI] | dz | % dialogs diff<0 | p (Holm) |",
              "|---|---|---|---|---|---|---|"]
    for _, x in g.iterrows():
        star = " **✓**" if x.significant else ""
        lines.append(f"| {x.contrast} | {x.metric} | {x.ratio} | {fmt(x.mean_diff)} [{fmt(x.ci95_lo)}, {fmt(x.ci95_hi)}] "
                     f"| {x.dz:+.2f} | {100 * x.pct_neg:.0f} | {x.p_holm:.4f}{star} |")
    lines.append("")
lines += ["## H4a: int8 KV is equivalent to bf16 KV (full context)", "",
          f"Equivalent if the 90% CI lies inside ±{EQ_MARGIN} nats (TOST, α=.05).", "",
          "| metric | silence | mean diff [90% CI] | equivalent |", "|---|---|---|---|"]
for _, x in eq.iterrows():
    lines.append(f"| {x.metric} | {x.ratio} | {fmt(x.mean_diff)} [{fmt(x.ci90_lo)}, {fmt(x.ci90_hi)}] | {'yes' if x.equivalent else 'no'} |")
lines += ["", "H5 (concurrent sessions under the 80 ms deadline) requires the serving measurement on the 3090."]
open(f"{out}/hypotheses.md", "w").write("\n".join(lines) + "\n")

# console summary
summ = res.groupby(["family", "metric"]).agg(tests=("p", "size"), significant=("significant", "sum"),
                                             mean_diff_range=("mean_diff", lambda s: f"{s.min():+.3f}..{s.max():+.3f}"))
print(summ.to_string())
print(eq[["metric", "ratio", "mean_diff", "ci90_lo", "ci90_hi", "equivalent"]].round(4).to_string(index=False))
print(f"report: {out}/hypotheses.md")
