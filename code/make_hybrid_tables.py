#!/usr/bin/env python3
# =============================================================================
# Tables + statistics for the CADENCE-H experiment (hybrid_results.csv)
#
#   python3 make_hybrid_tables.py --results hybrid_results.csv --outdir tables_hybrid
#
# Writes (LaTeX, \new{}-wrapped so they slot into the manuscript):
#   tab_hybrid_clf1.tex        per-dataset CL-F1, two regimes, mean +/- std
#   tab_hybrid_sig.tex         key contrasts: dataset-level (primary) + seed-level
#   tab_hybrid_retention.tex   retention on protected vs independent set + acceptance
#   tab_hybrid_gate.tex        per-dataset gate weight and per-expert F1
#   stats_summary.txt          everything printed, for the response letter
# =============================================================================
import argparse, os
import numpy as np, pandas as pd
from scipy import stats

DR = ["CICIDS2017", "UNSW_NB15", "IoT23", "MQTTset"]
CI = ["TON_IoT", "Edge_IIoTset"]
NICE = {"CICIDS2017": "CICIDS2017", "UNSW_NB15": "UNSW-NB15", "IoT23": "IoT-23",
        "MQTTset": "MQTTset", "TON_IoT": "TON-IoT", "Edge_IIoTset": "Edge-IIoTset"}
ORDER = ["Static-XGB", "Naive-XGB", "CADENCE-B", "ER-XGB", "CADENCE-R", "CADENCE-R-leaky", "ARF", "PWPAE-fair",
         "PWPAE-interleaved", "DWM", "CADENCE-H-ng", "CADENCE-H"]
FIXW = [round(0.1 * i, 1) for i in range(11)]


def per_run(df, col):
    """(method, dataset, seed) -> mean over stages/subs."""
    return df.groupby(["method", "dataset", "seed"])[col].mean().reset_index()


def contrast(pr, col, a, b, subset):
    p = pr[pr.dataset.isin(subset)].pivot_table(index=["dataset", "seed"], columns="method", values=col)
    if a not in p or b not in p:
        return None
    A, B = p[a].values, p[b].values
    ds = p.groupby("dataset").mean()
    dA, dB = ds[a].values, ds[b].values
    d = dA - dB
    out = dict(delta=float(np.mean(d)), pos=int((d > 0).sum()), n=len(d))
    out["ds_t"] = stats.ttest_rel(dA, dB).pvalue if len(d) > 1 else np.nan
    try:
        out["ds_w"] = stats.wilcoxon(dA, dB).pvalue if len(d) > 1 else np.nan
    except Exception:
        out["ds_w"] = np.nan
    if len(d) > 1:
        ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
        out["ci"] = (float(ci[0]), float(ci[1]))
        out["dz"] = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else np.nan
    else:
        out["ci"] = (np.nan, np.nan); out["dz"] = np.nan
    out["seed_t"] = stats.ttest_rel(A, B).pvalue
    try:
        out["seed_w"] = stats.wilcoxon(A, B).pvalue
    except Exception:
        out["seed_w"] = np.nan
    out["n_seed"] = len(A)
    return out


def fmt_p(p):
    if p is None or np.isnan(p):
        return "--"
    return "$<0.001$" if p < 0.001 else "$%.3f$" % p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="hybrid_results.csv")
    ap.add_argument("--outdir", default="tables_hybrid")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    df = pd.read_csv(a.results)
    methods = [m for m in ORDER if m in df.method.unique()]
    log = []

    def P(*s):
        line = " ".join(str(x) for x in s); print(line); log.append(line)

    # ------------------------------------------------------------------ T1
    pr = per_run(df, "f1")
    g = pr.groupby(["dataset", "method"]).f1.agg(["mean", "std"])
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{\new{Prequential CL-F1 of the validation-gated mixture (CADENCE-H) and all comparison arms "
         r"(mean\,$\pm$\,standard deviation over three seeds). Every method, including the online experts, scores "
         r"each chunk in full before receiving its labels; the guard set is never seen by a batch candidate. "
         r"Best per row in bold.}}", r"\label{tab:hybrid_clf1}", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
         r"\begin{tabular}{l" + "c" * len(methods) + "}", r"\toprule",
         r"\new{Dataset} & " + " & ".join(r"\new{%s}" % m for m in methods) + r" \\"]
    for lab, grp in [("(a) Time-interleaved distribution-shift streams", DR), ("(b) Class-incremental streams", CI)]:
        L += [r"\midrule", r"\multicolumn{%d}{l}{\new{\emph{%s}}} \\" % (len(methods) + 1, lab), r"\midrule"]
        means = {m: [] for m in methods}
        for ds in grp:
            if ds not in df.dataset.unique():
                continue
            vals = {m: (g.loc[(ds, m), "mean"] if (ds, m) in g.index else np.nan) for m in methods}
            best = np.nanmax(list(vals.values()))
            cells = []
            for m in methods:
                v = vals[m]
                if np.isnan(v):
                    cells.append(r"\new{--}"); continue
                means[m].append(v)
                sd = g.loc[(ds, m), "std"]
                s = r"\textbf{%.4f}" % v if abs(v - best) < 1e-9 else "%.4f" % v
                cells.append(r"\new{%s{\scriptsize\,$\pm$%.3f}}" % (s, sd))
            L.append(r"\new{%s} & " % NICE[ds] + " & ".join(cells) + r" \\")
        mv = {m: (np.mean(means[m]) if means[m] else np.nan) for m in methods}
        b = np.nanmax(list(mv.values()))
        L.append(r"\cmidrule(l){1-%d}" % (len(methods) + 1))
        L.append(r"\new{\textbf{Mean}} & " + " & ".join(
            r"\new{%s}" % (r"\textbf{%.4f}" % v if abs(v - b) < 1e-9 else "%.4f" % v) if not np.isnan(v) else r"\new{--}"
            for v in mv.values()) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    open(os.path.join(a.outdir, "tab_hybrid_clf1.tex"), "w").write("\n".join(L))

    P("=" * 70); P("CL-F1 per dataset"); P("=" * 70)
    P(pr.pivot_table(index="dataset", columns="method", values="f1").round(4).to_string())
    P("\nDRIFT-4 mean :"); P(pr[pr.dataset.isin(DR)].groupby("method").f1.mean().round(4).sort_values(ascending=False).to_string())
    P("\nCLASS-INCR mean :"); P(pr[pr.dataset.isin(CI)].groupby("method").f1.mean().round(4).sort_values(ascending=False).to_string())

    # ------------------------------------------------------------------ T2
    contrasts = [
        ("CADENCE-H", "CADENCE-R", "mixture vs batch expert alone"),
        ("CADENCE-H", "ARF", "mixture vs online expert alone"),
        ("CADENCE-H", "PWPAE-fair", "mixture vs PWPAE (fair timing)"),
        ("CADENCE-H", "Static-XGB", "mixture vs static"),
        ("CADENCE-B", "Naive-XGB", "admission on batch-only candidates (matched)"),
        ("ER-XGB", "Naive-XGB", "replay at matched update policy"),
        ("CADENCE-H", "CADENCE-H-ng", "admission inside the mixture (matched)"),
        ("PWPAE-interleaved", "PWPAE-fair", "label-release schedule (matched)"),
        ("CADENCE-R-leaky", "CADENCE-R", "guard overlap with training records (matched)"),
        ("CADENCE-H", "DWM", "prequential gate vs Dynamic Weighted Majority (matched)"),
        ("CADENCE-B", "Static-XGB", "batch-only guarded adaptation vs static (package)"),
        ("CADENCE-R", "ER-XGB", "guard on top of replay"),
        ("ER-XGB", "Static-XGB", "replay vs static"),
    ]
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{\new{Statistical comparison under the primary protocol on the four time-interleaved streams. "
         r"Dataset-level tests ($n{=}4$, seeds averaged first) are primary; seed-level tests ($n{=}12$) are "
         r"secondary. `Pos.' counts streams on which the first method is ahead.}}", r"\label{tab:hybrid_sig}",
         r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\resizebox{\columnwidth}{!}{%",
         r"\begin{tabular}{lcccccc}", r"\toprule",
         r"& & \multicolumn{3}{c}{\new{Dataset-level ($n{=}4$)}} & \multicolumn{2}{c}{\new{Seed-level ($n{=}12$)}} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}",
         r"\new{Comparison} & \new{$\Delta$} & \new{Pos.} & \new{$t$} & \new{95\% CI} & \new{$t$} & \new{Wilcoxon} \\",
         r"\midrule"]
    P("\n" + "=" * 70); P("KEY CONTRASTS — drift-4, dataset-level primary"); P("=" * 70)
    holm = []
    for x, y, lab in contrasts:
        r = contrast(pr, "f1", x, y, DR)
        if r is None:
            continue
        holm.append((x, y, r["ds_t"]))
        P("  %-12s vs %-12s d=%+.4f  pos %d/%d  ds_t=%.3f ds_W=%.3f  CI[%+.4f,%+.4f]  d_z=%.2f | seed_t=%.4f seed_W=%.4f   <- %s"
          % (x, y, r["delta"], r["pos"], r["n"], r["ds_t"], r["ds_w"], r["ci"][0], r["ci"][1], r["dz"], r["seed_t"], r["seed_w"], lab))
        L.append(r"\new{%s vs %s} & \new{$%+.4f$} & \new{%d/%d} & \new{%s} & \new{$[%+.3f,\,%+.3f]$} & \new{%s} & \new{%s} \\"
                 % (x, y, r["delta"], r["pos"], r["n"], fmt_p(r["ds_t"]), r["ci"][0], r["ci"][1], fmt_p(r["seed_t"]), fmt_p(r["seed_w"])))
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    open(os.path.join(a.outdir, "tab_hybrid_sig.tex"), "w").write("\n".join(L))

    # ------------------------------------------------------------------ T3
    rv = per_run(df, "ret_val"); rh = per_run(df, "ret_hist")
    acc = df[df.accepted.notna() & df.dataset.isin(DR)].copy()      # SAME scope as the retention numbers
    acc["a"] = acc.accepted.map({True: 1, False: 0, "True": 1, "False": 0})
    ag = acc.groupby("method").a.agg(["sum", "count"])
    ar = ag["sum"] / ag["count"] * 100
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{\new{Historical retention on the drifting streams measured on the protected calibration split "
         r"($\mathcal{X}_{\mathrm{val}}$) and on an independent historical test set ($\mathcal{X}_{\mathrm{hist}}$) that is "
         r"never used for training, calibration or admission; and the fraction of candidate updates admitted by a "
         r"leakage-free guard, shown as admitted/attempted on these four streams.}}", r"\label{tab:hybrid_retention}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lcccc}", r"\toprule",
         r"\new{Method} & \new{Ret.\ (val)} & \new{Ret.\ (hist)} & \new{Inflation} & \new{Accepted} \\", r"\midrule"]
    P("\n" + "=" * 70); P("RETENTION (drift-4): protected vs independent"); P("=" * 70)
    for m in methods:
        v = rv[(rv.method == m) & rv.dataset.isin(DR)].ret_val.mean()
        h = rh[(rh.method == m) & rh.dataset.isin(DR)].ret_hist.mean()
        acs = ("%d/%d (%.1f\\%%)" % (ag.loc[m, "sum"], ag.loc[m, "count"], ar[m])) if m in ar.index else "--"
        P("  %-12s ret_val=%.4f ret_hist=%.4f inflation=%+.4f accepted=%s" % (m, v, h, v - h, acs.replace("\\", "")))
        L.append(r"\new{%s} & \new{%.4f} & \new{%.4f} & \new{$%+.4f$} & \new{%s} \\" % (m, v, h, v - h, acs))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(a.outdir, "tab_hybrid_retention.tex"), "w").write("\n".join(L))

    # ------------------------------------------------------------------ T4
    if "CADENCE-H" in methods:
        h = df[df.method == "CADENCE-H"]
        gg = h.groupby("dataset")[["w", "f1_B", "f1_O", "f1"]].mean()
        L = [r"\begin{table}[t]", r"\centering",
             r"\caption{\new{Behaviour of the validation gate. Mean trust weight $w_t$ assigned to the batch expert, "
             r"and the CL-F1 each expert would attain alone versus the mixture, per stream. Best of the three in bold.}}",
             r"\label{tab:hybrid_gate}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lcccc}", r"\toprule",
             r"\new{Dataset} & \new{mean $w_t$} & \new{$F_B$ alone} & \new{$F_O$ alone} & \new{Mixture} \\", r"\midrule"]
        P("\n" + "=" * 70); P("GATE BEHAVIOUR"); P("=" * 70)
        for ds in DR + CI:
            if ds not in gg.index:
                continue
            r = gg.loc[ds]
            P("  %-13s w=%.2f  F_B=%.4f  F_O=%.4f  mix=%.4f  (mix - best expert = %+.4f)"
              % (ds, r.w, r.f1_B, r.f1_O, r.f1, r.f1 - max(r.f1_B, r.f1_O)))
            vals = [r.f1_B, r.f1_O, r.f1]; best = max(vals)
            cells = [("\\textbf{%.4f}" % v) if abs(v - best) < 1e-9 else ("%.4f" % v) for v in vals]
            L.append(r"\new{%s} & \new{%.2f} & \new{%s} & \new{%s} & \new{%s} \\" % (NICE[ds], r.w, cells[0], cells[1], cells[2]))
        L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        open(os.path.join(a.outdir, "tab_hybrid_gate.tex"), "w").write("\n".join(L))

    # ------------------------------------------------------------------ T5 (gate controls; only if logged)
    if "CADENCE-H" in methods and "f1_fix50" in df.columns:
        h = df[df.method == "CADENCE-H"]
        pc = h.groupby(["dataset", "seed"])[["f1", "f1_fix50", "f1_fixcal", "f1_hard"]].mean().reset_index()
        if "CADENCE-H-ng" in methods:
            ng = df[df.method == "CADENCE-H-ng"].groupby(["dataset", "seed"]).f1.mean().reset_index().rename(columns={"f1": "f1_ng"})
            pc = pc.merge(ng, on=["dataset", "seed"], how="left")
        cols = [("f1", "CADENCE-H (adaptive $w_t$)"), ("f1_fix50", "fixed $w=0.5$"),
                ("f1_fixcal", "fixed $w=w_0$ (calibration)"), ("f1_hard", "hard selection (previous stage)")]
        if "f1_ng" in pc.columns:
            cols.append(("f1_ng", "CADENCE-H without admission"))
        L = [r"\begin{table}[t]", r"\centering",
             r"\caption{\new{Matched gate controls. The three fixed/selection rows are computed from the \emph{same} saved expert scores as CADENCE-H; the no-admission row is a separate arm. Dataset-level paired differences of CADENCE-H minus the control on the four drift streams; `Pos.' counts streams on which CADENCE-H is ahead.}}",
             r"\label{tab:hybrid_controls}", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{lccccc}", r"\toprule",
             r"\new{Predictor} & \new{Drift-4} & \new{Class-incr.} & \new{H $-$ ctrl} & \new{Pos.} & \new{$t$ / W} \\", r"\midrule"]
        P("\n" + "=" * 70); P("GATE CONTROLS (same saved scores)"); P("=" * 70)
        base = pc[pc.dataset.isin(DR)].groupby("dataset").f1.mean()
        for col, lab in cols:
            dr = pc[pc.dataset.isin(DR)].groupby("dataset")[col].mean()
            ci_ = pc[pc.dataset.isin(CI)].groupby("dataset")[col].mean()
            dd = (base - dr.reindex(base.index)).values
            if col == "f1":
                tp = wp = np.nan
            else:
                tp = stats.ttest_rel(base.values, dr.reindex(base.index).values).pvalue if dd.std() > 0 else np.nan
                try: wp = stats.wilcoxon(base.values, dr.reindex(base.index).values).pvalue if np.any(dd != 0) else np.nan
                except Exception: wp = np.nan
            P("  %-32s drift=%.4f  class-incr=%.4f  H-ctrl=%+.4f  pos=%d/%d  t=%.3f  W=%.3f" % (lab, dr.mean(), ci_.mean(), dd.mean(), int((dd > 0).sum()), len(dd), tp, wp))
            L.append(r"\new{%s} & \new{%.4f} & \new{%.4f} & \new{%s} & \new{%s} & \new{%s} \\" % (
                lab, dr.mean(), ci_.mean(), ("$%+.4f$" % dd.mean()) if col != "f1" else "--",
                ("%d/%d" % (int((dd > 0).sum()), len(dd))) if col != "f1" else "--",
                ("%s / %s" % (fmt_p(tp), fmt_p(wp))) if col != "f1" else "--"))
        L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        open(os.path.join(a.outdir, "tab_hybrid_controls.tex"), "w").write("\n".join(L))
        if "changed_vs_best" in df.columns:
            P("  fraction of records whose class differs from the better expert's: %.3f" % h.changed_vs_best.mean())

    # ------------------------------------------------------------------ Holm over the pre-specified family
    if holm:
        P("\n" + "=" * 70); P("HOLM-ADJUSTED dataset-level t p-values (drift-4, family = %d contrasts)" % len(holm)); P("=" * 70)
        ps = np.array([h[2] for h in holm]); order = np.argsort(ps); m = len(ps); adj = np.empty(m)
        run = 0.0
        for rank, i in enumerate(order):
            run = max(run, (m - rank) * ps[i]); adj[i] = min(1.0, run)
        for (x, y, pv), pa in zip(holm, adj):
            P("  %-18s vs %-18s raw=%.3f  holm=%.3f" % (x, y, pv, pa))

    # ------------------------------------------------------------------ T6 protocol A/Bs (v4)
    pairs = [("PWPAE-interleaved", "PWPAE-fair", "Label release: per-sample vs stage-level"),
             ("CADENCE-R-leaky", "CADENCE-R", "Admission slice: overlapping vs disjoint")]
    have = [p_ for p_ in pairs if p_[0] in methods and p_[1] in methods]
    if have:
        P("\n" + "=" * 70); P("PROTOCOL A/Bs (matched pairs)"); P("=" * 70)
        L = [r"\begin{table*}[t]", r"\centering",
             r"\caption{\new{Protocol confounds measured as matched pairs. Each pair differs in exactly one respect. "
             r"Label release: the same PWPAE members scored per sample with interleaved learning versus scoring the whole stage before any member learns. "
             r"Admission slice: the same rehearsal candidate admitted on a slice drawn from its own training records versus the disjoint guard partition. "
             r"CL-F1 per stream (mean over seeds), the paired difference, and acceptance where applicable.}}",
             r"\label{tab:hybrid_protocol}", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
             r"\begin{tabular}{ll" + "c" * 6 + "cc}", r"\toprule",
             r"\new{Pair} & \new{Arm} & " + " & ".join(r"\new{%s}" % NICE[d_] for d_ in DR + CI) + r" & \new{Drift-4} & \new{All-6} \\", r"\midrule"]
        for x, y, lab in have:
            for arm in (x, y):
                vals = [pr[(pr.method == arm) & (pr.dataset == d_)].f1.mean() for d_ in DR + CI]
                d4 = np.nanmean(vals[:4]); a6 = np.nanmean(vals)
                acc_ = df[(df.method == arm) & df.dataset.isin(DR) & df.accepted.notna()]
                accs = ("  acc=%d/%d" % (acc_.accepted.astype(float).sum(), len(acc_))) if len(acc_) else ""
                P("  %-40s %-18s " % (lab, arm) + " ".join("%.4f" % v for v in vals) + "  drift=%.4f all=%.4f%s" % (d4, a6, accs))
                L.append(r"\new{%s} & \new{%s} & " % (lab if arm == x else "", arm) + " & ".join(r"\new{%.4f}" % v for v in vals) + r" & \new{%.4f} & \new{%.4f} \\" % (d4, a6))
            r4 = contrast(pr, "f1", x, y, DR); r6 = contrast(pr, "f1", x, y, DR + CI)
            P("     delta drift-4 %+.4f pos %d/%d t=%.3f CI[%+.3f,%+.3f] | all-6 %+.4f pos %d/%d t=%.3f W=%.3f" % (
                r4["delta"], r4["pos"], r4["n"], r4["ds_t"], r4["ci"][0], r4["ci"][1], r6["delta"], r6["pos"], r6["n"], r6["ds_t"], r6["ds_w"]))
            L.append(r"\new{} & \new{\emph{difference}} & " + " & ".join(
                r"\new{$%+.3f$}" % (pr[(pr.method == x) & (pr.dataset == d_)].f1.mean() - pr[(pr.method == y) & (pr.dataset == d_)].f1.mean()) for d_ in DR + CI)
                + r" & \new{$%+.4f$ (%d/%d, $p{=}%s$)} & \new{$%+.4f$ (%d/%d, $p{=}%s$)} \\" % (
                    r4["delta"], r4["pos"], r4["n"], fmt_p(r4["ds_t"]).strip("$"), r6["delta"], r6["pos"], r6["n"], fmt_p(r6["ds_t"]).strip("$")))
            L.append(r"\midrule")
        L[-1] = r"\bottomrule"; L += [r"\end{tabular}", r"\end{table*}"]
        open(os.path.join(a.outdir, "tab_hybrid_protocol.tex"), "w").write("\n".join(L))

    # ------------------------------------------------------------------ T7 combination-rule benchmark incl. fixed-w sweep (v4)
    if "CADENCE-H" in methods and "f1_fix00" in df.columns:
        h = df[df.method == "CADENCE-H"]
        sw_cols = ["f1_fix%02d" % int(round(fw * 10)) for fw in FIXW]
        pc = h.groupby(["dataset", "seed"])[["f1", "f1_hard", "f1_fixcal"] + sw_cols].mean().reset_index()
        if "DWM" in methods:
            dw = df[df.method == "DWM"].groupby(["dataset", "seed"]).f1.mean().reset_index().rename(columns={"f1": "f1_dwm"})
            pc = pc.merge(dw, on=["dataset", "seed"], how="left")
        dsm = pc.groupby("dataset").mean(numeric_only=True)
        rows_ = [("f1_fix%02d" % int(round(fw * 10)), "fixed $w=%.1f$" % fw + (" (online alone)" if fw == 0 else " (batch alone)" if fw == 1 else "")) for fw in FIXW]
        rows_ += [("f1_fixcal", "fixed $w=w_0$ (calibration)"), ("f1_hard", "hard selection (previous-stage winner)")]
        if "f1_dwm" in dsm.columns: rows_.append(("f1_dwm", "Dynamic Weighted Majority"))
        rows_.append(("f1", "prequential gate (CADENCE-H)"))
        M = dsm[[c for c, _ in rows_]]
        rk_all = M.rank(axis=1, ascending=False).mean(); rk_dr = M.loc[[d_ for d_ in DR if d_ in M.index]].rank(axis=1, ascending=False).mean()
        best_fix_dr = max(sw_cols, key=lambda c: M.loc[[d_ for d_ in DR if d_ in M.index], c].mean())
        best_fix_ci = max(sw_cols, key=lambda c: M.loc[[d_ for d_ in CI if d_ in M.index], c].mean())
        best_fix_all = max(sw_cols, key=lambda c: M[c].mean())
        P("\n" + "=" * 70); P("COMBINATION RULES over the same saved expert scores (fixed-w sweep, hard, DWM, gate)"); P("=" * 70)
        P("  best fixed w: drift-4 %s  class-incr %s  all-6 %s" % (best_fix_dr, best_fix_ci, best_fix_all))
        L = [r"\begin{table}[t]", r"\centering",
             r"\caption{\new{Combination rules over the same two experts. Fixed weights and hard selection are computed from the saved scores of the CADENCE-H run; Dynamic Weighted Majority is a separate arm over the same experts. Mean CL-F1 per regime and mean rank across streams (lower is better). Best per column in bold.}}",
             r"\label{tab:hybrid_rules}", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{lcccc}", r"\toprule",
             r"\new{Rule} & \new{Drift-4} & \new{Class-incr.} & \new{All-6} & \new{Mean rank} \\", r"\midrule"]
        col_vals = {}
        for c, lab in rows_:
            col_vals[c] = (M.loc[[d_ for d_ in DR if d_ in M.index], c].mean(), M.loc[[d_ for d_ in CI if d_ in M.index], c].mean(), M[c].mean(), rk_all[c])
        bests = [max(col_vals.values(), key=lambda v: v[i])[i] for i in range(3)] + [min(v[3] for v in col_vals.values())]
        for c, lab in rows_:
            v = col_vals[c]
            cells = [(r"\textbf{%.4f}" % v[i]) if abs(v[i] - bests[i]) < 1e-9 else ("%.4f" % v[i]) for i in range(3)]
            cells.append((r"\textbf{%.2f}" % v[3]) if abs(v[3] - bests[3]) < 1e-9 else ("%.2f" % v[3]))
            P("  %-42s drift=%.4f  ci=%.4f  all=%.4f  rank=%.2f" % (lab, *v))
            L.append(r"\new{%s} & " % lab + " & ".join(r"\new{%s}" % x_ for x_ in cells) + r" \\")
        L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        open(os.path.join(a.outdir, "tab_hybrid_rules.tex"), "w").write("\n".join(L))
        # gate vs BEST fixed w, dataset-level
        for scope, sub, bc in (("drift-4", DR, best_fix_dr), ("class-incr", CI, best_fix_ci), ("all-6", DR + CI, best_fix_all)):
            g = M.loc[[d_ for d_ in sub if d_ in M.index]]
            dd = (g["f1"] - g[bc]).values
            tp = stats.ttest_rel(g["f1"], g[bc]).pvalue if len(dd) > 1 and dd.std() > 0 else np.nan
            P("  gate - best fixed (%s, %s): %+.4f  pos %d/%d  t=%.3f" % (scope, bc, dd.mean(), int((dd > 0).sum()), len(dd), tp))

    # ------------------------------------------------------------------ T8 Friedman + average ranks over all arms (v4)
    ds_mean = pr.groupby(["dataset", "method"]).f1.mean().unstack("method")[methods]
    P("\n" + "=" * 70); P("FRIEDMAN + AVERAGE RANKS (dataset-level, all arms)"); P("=" * 70)
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{\new{Average rank of every arm across streams (1 = best), with the Friedman omnibus test over all arms. The Nemenyi critical difference is reported for reference; at this number of streams it does not separate individual pairs.}}",
         r"\label{tab:hybrid_ranks}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lcc}", r"\toprule",
         r"\new{Arm} & \new{Rank (drift-4)} & \new{Rank (all-6)} \\", r"\midrule"]
    rk4 = ds_mean.loc[[d_ for d_ in DR if d_ in ds_mean.index]].rank(axis=1, ascending=False).mean()
    rk6 = ds_mean.rank(axis=1, ascending=False).mean()
    for scope, sub in (("drift-4", DR), ("all-6", DR + CI)):
        g = ds_mean.loc[[d_ for d_ in sub if d_ in ds_mean.index]]
        if len(g) >= 3 and g.shape[1] >= 3:
            fr = stats.friedmanchisquare(*[g[m_].values for m_ in g.columns])
            k_, N_ = g.shape[1], len(g)
            # two-tailed Nemenyi q_0.05 (Demsar 2006, Table 5a), k = 2..20
            Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
                   11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544}
            CD = Q05[k_] * np.sqrt(k_ * (k_ + 1) / (6.0 * N_)) if k_ in Q05 else np.nan
            P("  Friedman %s: k=%d N=%d chi2=%.2f p=%.4f  (Nemenyi CD alpha=.05 = %.2f)" % (scope, k_, N_, fr.statistic, fr.pvalue, CD))
    for m_ in rk6.sort_values().index:
        P("  %-18s rank drift-4=%.2f  all-6=%.2f" % (m_, rk4.get(m_, np.nan), rk6[m_]))
        L.append(r"\new{%s} & \new{%.2f} & \new{%.2f} \\" % (m_, rk4.get(m_, np.nan), rk6[m_]))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(a.outdir, "tab_hybrid_ranks.tex"), "w").write("\n".join(L))

    # ------------------------------------------------------------------ stage-level mixed model (secondary), if statsmodels present
    try:
        import statsmodels.formula.api as smf
        P("\n" + "=" * 70); P("STAGE-LEVEL MIXED MODEL  f1 ~ arm + (1 | dataset:seed)   [secondary]"); P("=" * 70)
        for x, y, lab in contrasts:
            if x not in methods or y not in methods: continue
            for scope, sub in (("drift-4", DR), ("all-6", DR + CI)):
                s_ = df[df.dataset.isin(sub) & df.method.isin([x, y])].copy()
                s_["grp"] = s_.dataset + "_" + s_.seed.astype(str); s_["arm"] = (s_.method == x).astype(int)
                try:
                    mdl = smf.mixedlm("f1 ~ arm", s_, groups=s_["grp"]).fit(reml=True)
                    P("  %-18s vs %-18s %-8s coef=%+.4f  p=%.4f" % (x, y, scope, mdl.params["arm"], mdl.pvalues["arm"]))
                except Exception as e:
                    P("  %-18s vs %-18s %-8s LMM failed: %s" % (x, y, scope, e))
    except ImportError:
        P("\n(statsmodels not installed: stage-level mixed model skipped; pip3 install --user statsmodels)")

    # ------------------------------------------------------------------ T9 cost + T10 diagnostics (v4)
    if "sec_fit" in df.columns:
        cst = df[df.dataset.isin(DR)].groupby("method")[["sec_score", "sec_fit"]].mean().reindex(methods)
        P("\n" + "=" * 70); P("COST per stage (drift-4 mean, seconds): score / fit+update"); P("=" * 70)
        L = [r"\begin{table}[t]", r"\centering",
             r"\caption{\new{Mean wall-clock seconds per stage on the drift streams: scoring the stage, and fitting/updating after labels (single Linux node, one process).}}",
             r"\label{tab:hybrid_cost}", r"\footnotesize", r"\begin{tabular}{lcc}", r"\toprule", r"\new{Arm} & \new{Score (s)} & \new{Update (s)} \\", r"\midrule"]
        for m_ in methods:
            P("  %-18s score=%.2f  update=%.2f" % (m_, cst.loc[m_, "sec_score"], cst.loc[m_, "sec_fit"]))
            L.append(r"\new{%s} & \new{%.2f} & \new{%.2f} \\" % (m_, cst.loc[m_, "sec_score"], cst.loc[m_, "sec_fit"]))
        L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        open(os.path.join(a.outdir, "tab_hybrid_cost.tex"), "w").write("\n".join(L))
    if "n_guard" in df.columns and "CADENCE-H" in methods:
        h = df[df.method == "CADENCE-H"]
        dg = h.groupby("dataset").agg(n=("n", "mean"), n_guard=("n_guard", "mean"), gmin=("guard_min_class", "mean"),
                                      disagree=("disagree", "mean"), union=("oracle_union", "mean"), f1B=("f1_B", "mean"), f1O=("f1_O", "mean"), f1=("f1", "mean"))
        P("\n" + "=" * 70); P("STAGE / GUARD SIZES and EXPERT COMPLEMENTARITY (CADENCE-H rows)"); P("=" * 70)
        L = [r"\begin{table}[t]", r"\centering",
             r"\caption{\new{Records per stage and per guard slice (mean over seeds), the smallest per-class count in the guard slice, and expert complementarity: the fraction of stage records on which the two experts' predictions differ, and the fraction that at least one expert classifies correctly (record-level oracle).}}",
             r"\label{tab:hybrid_diag}", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{lcccccc}", r"\toprule",
             r"\new{Dataset} & \new{Stage} & \new{Guard} & \new{Min/class} & \new{Disagree} & \new{Oracle$_\cup$} & \new{Mixture} \\", r"\midrule"]
        for d_ in DR + CI:
            if d_ not in dg.index: continue
            r_ = dg.loc[d_]
            P("  %-13s stage=%4.0f guard=%3.0f min/class=%4.1f disagree=%.3f union=%.3f f1B=%.3f f1O=%.3f mix=%.3f" % (d_, r_.n, r_.n_guard, r_.gmin, r_.disagree, r_.union, r_.f1B, r_.f1O, r_.f1))
            L.append(r"\new{%s} & \new{%.0f} & \new{%.0f} & \new{%.1f} & \new{%.3f} & \new{%.3f} & \new{%.4f} \\" % (NICE[d_], r_.n, r_.n_guard, r_.gmin, r_.disagree, r_.union, r_.f1))
        L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        open(os.path.join(a.outdir, "tab_hybrid_diag.tex"), "w").write("\n".join(L))

    open(os.path.join(a.outdir, "stats_summary.txt"), "w").write("\n".join(log))
    print("\n[written]", a.outdir)


if __name__ == "__main__":
    main()
