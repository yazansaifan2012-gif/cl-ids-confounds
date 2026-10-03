#!/usr/bin/env python3
# Rerun-2 tables:
#   tab_hybrid_chrono.tex : CICIDS2017 file-order vs chronological, every arm, CL-F1 + rank, plus expert ordering & gate
#   tab_hybrid_srp.tex    : ARF-based vs SRP-based online expert: F_O alone, mixture, DWM; best fixed w vs gate, per regime
#   rerun2_summary.txt    : everything printed
import argparse, os
import numpy as np, pandas as pd
from scipy import stats
ORDER = ["Static-XGB", "Naive-XGB", "CADENCE-B", "ER-XGB", "CADENCE-R", "CADENCE-R-leaky", "ARF", "PWPAE-fair",
         "PWPAE-interleaved", "DWM", "CADENCE-H-ng", "CADENCE-H"]
DR = ["CICIDS2017", "UNSW_NB15", "IoT23", "MQTTset"]; CI = ["TON_IoT", "Edge_IIoTset"]
NICE = {"CICIDS2017": "CICIDS2017", "UNSW_NB15": "UNSW-NB15", "IoT23": "IoT-23", "MQTTset": "MQTTset", "TON_IoT": "TON-IoT", "Edge_IIoTset": "Edge-IIoTset"}
FIX = ["f1_fix%02d" % i for i in range(11)]

def safe_read(path):
    """Return a DataFrame or None for a missing / empty / header-less file."""
    try:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return None
        df = pd.read_csv(path)
        return df if len(df) and "f1" in df.columns else None
    except Exception:
        return None

def per_ds(df, col="f1"):
    return df.groupby(["method", "dataset", "seed"])[col].mean().groupby(["method", "dataset"]).mean().unstack(0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file-order", default="hybrid_results.csv"); ap.add_argument("--chrono", default="hybrid_results_chrono.csv")
    ap.add_argument("--srp", default="hybrid_results_srp.csv"); ap.add_argument("--outdir", default="tables_hybrid"); a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True); log = []
    def P(*x):
        line = " ".join(str(y) for y in x); print(line); log.append(line)
    F = pd.read_csv(a.file_order)

    # ------------------------------------------------------------ chrono
    Cc = safe_read(a.chrono)
    if Cc is not None and "CICIDS2017_chrono" in set(Cc.dataset):
        f = per_ds(F).loc["CICIDS2017"]; c = per_ds(Cc).loc["CICIDS2017_chrono"]
        methods = [m for m in ORDER if m in f.index and m in c.index]
        rf = f[methods].rank(ascending=False); rc = c[methods].rank(ascending=False)
        P("=" * 70); P("CICIDS2017: FILE-ORDER vs CHRONOLOGICAL (mean CL-F1 over seeds)"); P("=" * 70)
        L = [r"\begin{table}[t]", r"\centering",
             r"\caption{Stream-construction robustness on CICIDS2017: CL-F1 of every arm when the cleaned distribution is streamed in released-file order (F) and when the raw per-day flows are streamed in timestamp order (T), with each arm's rank within the stream (1 = best). Same class-floor budget, seeds and configuration; the two files differ in feature set and in row order.}",
             r"\label{tab:hybrid_chrono}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lcccc}", r"\toprule",
             r"Arm & F & T & rk$_F$ & rk$_T$ \\", r"\midrule"]
        for m in methods:
            P("  %-18s file=%.4f (rk %2d)   time=%.4f (rk %2d)" % (m, f[m], rf[m], c[m], rc[m]))
            L.append(r"%s & %.4f & %.4f & %d & %d \\" % (m, f[m], c[m], rf[m], rc[m]))
        L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        open(os.path.join(a.outdir, "tab_hybrid_chrono.tex"), "w").write("\n".join(L))
        rho = stats.spearmanr(f[methods], c[methods]).correlation
        P("  Spearman rank agreement file vs time: rho=%.3f ; best(F)=%s best(T)=%s" % (rho, f[methods].idxmax(), c[methods].idxmax()))
        for name, D, key in (("file ", F, "CICIDS2017"), ("time ", Cc, "CICIDS2017_chrono")):
            h = D[(D.method == "CADENCE-H") & (D.dataset == key)]
            g = h.groupby(["seed", "stage"])[["f1_B", "f1_O", "f1", "w"]].mean().groupby("stage").mean()
            P("  %s-order CADENCE-H per stage: F_B=%s  F_O=%s  w=%s" % (name, np.round(g.f1_B.values, 2).tolist(), np.round(g.f1_O.values, 2).tolist(), np.round(g.w.values, 2).tolist()))
            pi = D[(D.method == "PWPAE-interleaved") & (D.dataset == key)].f1.mean(); pf = D[(D.method == "PWPAE-fair") & (D.dataset == key)].f1.mean()
            P("  %s-order label-release effect: interleaved %.4f  fair %.4f  delta %+.4f" % (name, pi, pf, pi - pf))
            hh = h.groupby(["seed"])[FIX + ["f1"]].mean().mean()
            P("  %s-order best fixed w: %s=%.4f  gate=%.4f" % (name, hh[FIX].idxmax(), hh[FIX].max(), hh["f1"]))
    else:
        P("chrono results not found or empty: skipping tab_hybrid_chrono.tex")

    # ------------------------------------------------------------ SRP
    S = safe_read(a.srp)
    if S is not None:
        both = pd.concat([F[F.method.isin(["ARF", "CADENCE-H", "DWM", "CADENCE-R"])], S], ignore_index=True)
        pm = per_ds(both)
        pairs = [("ARF", "SRP", "online expert alone"), ("CADENCE-H", "CADENCE-H-srp", "prequential mixture"), ("DWM", "DWM-srp", "Dynamic Weighted Majority")]
        P("\n" + "=" * 70); P("ONLINE BACKBONE: ARF vs SRP"); P("=" * 70)
        L = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Second online backbone. Each row pair is the same arm with Adaptive Random Forest (ARF) or Streaming Random Patches (SRP) as the online expert $F_O$; the batch expert, guard, gate and seeds are unchanged. Mean CL-F1 per stream and per regime, and the best fixed weight versus the adaptive gate for each backbone.}",
             r"\label{tab:hybrid_srp}", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{llcccccccc}", r"\toprule",
             r"Arm & $F_O$ & " + " & ".join(NICE[d] for d in DR + CI) + r" & Drift-4 & All-6 \\", r"\midrule"]
        for x, y, lab in pairs:
            for arm, kind in ((x, "ARF"), (y, "SRP")):
                if arm not in pm.columns: continue
                v = [pm.loc[d, arm] if d in pm.index else np.nan for d in DR + CI]
                d4 = np.nanmean(v[:4]); a6 = np.nanmean(v)
                P("  %-28s %-4s " % (lab, kind) + " ".join("%.4f" % q for q in v) + "  drift=%.4f all=%.4f" % (d4, a6))
                L.append(r"%s & %s & " % (lab if kind == "ARF" else "", kind) + " & ".join("%.4f" % q for q in v) + r" & %.4f & %.4f \\" % (d4, a6))
            L.append(r"\midrule")
        # fixed-w sweep vs gate under SRP
        for arm, kind in (("CADENCE-H", "ARF"), ("CADENCE-H-srp", "SRP")):
            D = F if kind == "ARF" else S
            h = D[D.method == arm]
            if h.empty or "f1_fix00" not in h.columns: continue
            g = h.groupby(["dataset", "seed"])[FIX + ["f1"]].mean().groupby("dataset").mean()
            for scope, sub in (("drift-4", DR), ("class-incr", CI), ("all-6", DR + CI)):
                gg = g.loc[[d for d in sub if d in g.index]]
                bc = gg[FIX].mean().idxmax()
                P("  %s gate vs best fixed (%s): gate=%.4f  best=%s %.4f  gate-best=%+.4f  pos %d/%d" % (kind, scope, gg.f1.mean(), bc, gg[bc].mean(), gg.f1.mean() - gg[bc].mean(), int((gg.f1 > gg[bc]).sum()), len(gg)))
                if scope != "class-incr":
                    L.append(r"\multicolumn{10}{l}{\emph{%s, %s: gate %.4f; best fixed $w$ = %s, %.4f; gate $-$ best $= %+.4f$}} \\" % (kind, scope, gg.f1.mean(), "%.1f" % (int(bc[-2:]) / 10.0), gg[bc].mean(), gg.f1.mean() - gg[bc].mean()))
        if L[-1] == r"\midrule": L.pop()
        L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
        open(os.path.join(a.outdir, "tab_hybrid_srp.tex"), "w").write("\n".join(L))
        # gate vs DWM under SRP
        if "CADENCE-H-srp" in pm.columns and "DWM-srp" in pm.columns:
            x = pm["CADENCE-H-srp"] - pm["DWM-srp"]
            P("  SRP: gate - DWM all-6 %+.4f  pos %d/%d" % (x.mean(), int((x > 0).sum()), len(x)))
    else:
        P("SRP results not found or empty: skipping tab_hybrid_srp.tex")
    open(os.path.join(a.outdir, "rerun2_summary.txt"), "w").write("\n".join(log)); print("[written]", a.outdir)

if __name__ == "__main__":
    main()
