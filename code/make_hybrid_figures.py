#!/usr/bin/env python3
# =============================================================================
# Figures for the CADENCE-H experiment (hybrid_results.csv)
#
#   python3 make_hybrid_figures.py --results hybrid_results.csv --outdir figs_hybrid
#
# Produces (PNG, 300 dpi, IEEE two-column friendly):
#   fig_hybrid_experts.png   per-stage F1: mixture vs batch expert vs online expert, 6 streams
#   fig_hybrid_gate.png      trust weight w_t per stage, 6 streams (shows the gate acting)
#   fig_hybrid_compare.png   drift-subset mean CL-F1, all methods, with seed spread
#   fig_hybrid_retention.png retention on protected vs independent set, all methods
# =============================================================================
import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DR = ["CICIDS2017", "UNSW_NB15", "IoT23", "MQTTset"]
CI = ["TON_IoT", "Edge_IIoTset"]
NICE = {"CICIDS2017": "CICIDS2017", "UNSW_NB15": "UNSW-NB15", "IoT23": "IoT-23",
        "MQTTset": "MQTTset", "TON_IoT": "TON-IoT", "Edge_IIoTset": "Edge-IIoTset"}
ORDER = ["Static-XGB", "Naive-XGB", "CADENCE-B", "ER-XGB", "CADENCE-R", "CADENCE-R-leaky", "ARF", "PWPAE-fair",
         "PWPAE-interleaved", "DWM", "CADENCE-H-ng", "CADENCE-H"]
COL = {"Static-XGB": "#7f7f7f", "Naive-XGB": "#c7c7c7", "CADENCE-B": "#9ecae1", "ER-XGB": "#fdae6b",
       "CADENCE-R": "#3182bd", "CADENCE-R-leaky": "#08519c", "ARF": "#74c476", "PWPAE-fair": "#31a354",
       "PWPAE-interleaved": "#006d2c", "DWM": "#9e9ac8", "CADENCE-H-ng": "#fbb4ae", "CADENCE-H": "#d62728"}

plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "figure.dpi": 300, "savefig.dpi": 300, "axes.spines.top": False,
                     "axes.spines.right": False})


def stage_curve(df, method, ds, col):
    """mean and std over seeds of per-stage value (subs averaged into stage)."""
    s = df[(df.method == method) & (df.dataset == ds)]
    if s.empty:
        return None, None, None
    per = s.groupby(["seed", "stage"])[col].mean().reset_index()
    g = per.groupby("stage")[col].agg(["mean", "std"])
    return g.index.values, g["mean"].values, g["std"].fillna(0).values


def fig_experts(df, out):
    if "CADENCE-H" not in df.method.unique():
        return
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2))
    for ax, ds in zip(axes.ravel(), DR + CI):
        h = df[(df.method == "CADENCE-H") & (df.dataset == ds)]
        if h.empty:
            ax.set_visible(False); continue
        for col, lab, c, ls in [("f1_B", r"$F_B$ (batch) alone", COL["CADENCE-R"], "--"),
                                ("f1_O", r"$F_O$ (online) alone", COL["ARF"], ":"),
                                ("f1", "CADENCE-H mixture", COL["CADENCE-H"], "-")]:
            x, m, sd = stage_curve(h, "CADENCE-H", ds, col)
            ax.plot(x, m, ls, color=c, lw=1.6, label=lab)
            ax.fill_between(x, m - sd, m + sd, color=c, alpha=0.12, lw=0)
        ax.set_title(NICE[ds] + ("  (class-incr.)" if ds in CI else ""))
        ax.set_xlabel("stage"); ax.set_ylim(0, 1.0); ax.set_xticks(range(8))
    axes[0, 0].set_ylabel("macro-F1"); axes[1, 0].set_ylabel("macro-F1")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(out, "fig_hybrid_experts.png"), bbox_inches="tight"); plt.close(fig)


def fig_gate(df, out):
    if "CADENCE-H" not in df.method.unique():
        return
    fig, axes = plt.subplots(1, 6, figsize=(7.2, 1.7), sharey=True)
    for ax, ds in zip(axes, DR + CI):
        h = df[(df.method == "CADENCE-H") & (df.dataset == ds)]
        if h.empty:
            ax.set_visible(False); continue
        x, m, sd = stage_curve(h, "CADENCE-H", ds, "w")
        ax.plot(x, m, "-o", color=COL["CADENCE-H"], lw=1.4, ms=2.5)
        ax.fill_between(x, m - sd, m + sd, color=COL["CADENCE-H"], alpha=0.15, lw=0)
        ax.axhline(0.5, color="k", lw=0.5, ls=":")
        ax.set_title(NICE[ds]); ax.set_ylim(0, 1); ax.set_xticks([0, 3, 7]); ax.set_xlabel("stage")
    axes[0].set_ylabel(r"trust in $F_B$  ($w_t$)")
    fig.text(0.5, 1.02, r"$w_t\to1$: batch expert trusted;  $w_t\to0$: online expert trusted", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_hybrid_gate.png"), bbox_inches="tight"); plt.close(fig)


def fig_compare(df, out):
    per = df.groupby(["method", "dataset", "seed"]).f1.mean().reset_index()
    methods = [m for m in ORDER if m in per.method.unique()]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1, 1]})
    for ax, (lab, grp) in zip(axes, [("Time-interleaved streams (drift)", DR), ("Class-incremental streams", CI)]):
        sub = per[per.dataset.isin(grp)]
        ds_mean = sub.groupby(["method", "dataset"]).f1.mean().reset_index()
        means = ds_mean.groupby("method").f1.mean().reindex(methods)
        # spread: across datasets
        lo = ds_mean.groupby("method").f1.min().reindex(methods)
        hi = ds_mean.groupby("method").f1.max().reindex(methods)
        xs = np.arange(len(methods))
        ax.bar(xs, means.values, color=[COL[m] for m in methods], width=0.7,
               edgecolor="k", linewidth=0.4)
        ax.errorbar(xs, means.values, yerr=[means.values - lo.values, hi.values - means.values],
                    fmt="none", ecolor="k", elinewidth=0.6, capsize=2)
        for x, v in zip(xs, means.values):
            ax.text(x, v + 0.015, "%.3f" % v, ha="center", va="bottom", fontsize=6)
        ax.set_xticks(xs); ax.set_xticklabels(methods, rotation=35, ha="right")
        ax.set_title(lab); ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("mean CL-F1 (bars)\nper-dataset range (whiskers)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_hybrid_compare.png"), bbox_inches="tight"); plt.close(fig)


def fig_retention(df, out):
    per = df[df.dataset.isin(DR)].groupby(["method", "dataset", "seed"])[["ret_val", "ret_hist"]].mean().reset_index()
    methods = [m for m in ORDER if m in per.method.unique()]
    g = per.groupby("method")[["ret_val", "ret_hist"]].mean().reindex(methods)
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    xs = np.arange(len(methods)); w = 0.38
    ax.bar(xs - w / 2, g.ret_val.values, w, color="#bdbdbd", edgecolor="k", linewidth=0.4,
           label="protected calibration split")
    ax.bar(xs + w / 2, g.ret_hist.values, w, color="#636363", edgecolor="k", linewidth=0.4,
           label="independent historical set")
    ax.set_xticks(xs); ax.set_xticklabels(methods, rotation=35, ha="right")
    ax.set_ylabel("historical macro-F1"); ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_hybrid_retention.png"), bbox_inches="tight"); plt.close(fig)


def fig_sweep(df, out):
    if "CADENCE-H" not in df.method.unique() or "f1_fix00" not in df.columns:
        return
    h = df[df.method == "CADENCE-H"]
    cols = ["f1_fix%02d" % i for i in range(11)]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=False)
    for ax, (lab, grp) in zip(axes, [("Time-interleaved streams (drift)", DR), ("Class-incremental streams", CI)]):
        for ds in grp:
            g = h[h.dataset == ds].groupby(["seed"])[cols + ["f1", "f1_hard"]].mean().mean()
            ax.plot(np.arange(11) / 10, g[cols].values, "-o", ms=2.5, lw=1.2, label=NICE[ds])
            ax.axhline(g["f1"], ls="--", lw=0.8, color=ax.lines[-1].get_color())
        ax.set_xlabel(r"fixed weight $w$ on the batch expert"); ax.set_title(lab); ax.set_xticks([0, 0.5, 1])
        ax.legend(frameon=False, fontsize=6, ncol=2)
    axes[0].set_ylabel("CL-F1 (solid: fixed $w$; dashed: adaptive gate)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_hybrid_sweep.png"), bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="hybrid_results.csv")
    ap.add_argument("--outdir", default="figs_hybrid")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    df = pd.read_csv(a.results)
    fig_experts(df, a.outdir)
    fig_gate(df, a.outdir)
    fig_compare(df, a.outdir)
    fig_retention(df, a.outdir)
    fig_sweep(df, a.outdir)
    print("[written]", a.outdir, "->", sorted(os.listdir(a.outdir)))


if __name__ == "__main__":
    main()
