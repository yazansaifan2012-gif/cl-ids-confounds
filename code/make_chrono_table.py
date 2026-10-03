#!/usr/bin/env python3
# File-order vs chronological streams: per-arm CL-F1 on the datasets present in BOTH results files.
#   python3 make_chrono_table.py --file-order hybrid_results.csv --chrono hybrid_results_chrono.csv --outdir tables_hybrid
import argparse, os
import numpy as np, pandas as pd
from scipy import stats
ORDER = ["Static-XGB", "Naive-XGB", "CADENCE-B", "ER-XGB", "CADENCE-R", "CADENCE-R-leaky", "ARF", "PWPAE-fair",
         "PWPAE-interleaved", "DWM", "CADENCE-H-ng", "CADENCE-H"]
NICE = {"CICIDS2017": "CICIDS2017", "UNSW_NB15": "UNSW-NB15", "IoT23": "IoT-23", "MQTTset": "MQTTset", "TON_IoT": "TON-IoT", "Edge_IIoTset": "Edge-IIoTset"}

def per_ds(df):
    return df.groupby(["method", "dataset", "seed"]).f1.mean().groupby(["method", "dataset"]).mean().unstack(0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file-order", default="hybrid_results.csv"); ap.add_argument("--chrono", default="hybrid_results_chrono.csv")
    ap.add_argument("--outdir", default="tables_hybrid"); a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    F = pd.read_csv(a.file_order); Cc = pd.read_csv(a.chrono)
    ds_common = [d for d in ["CICIDS2017", "UNSW_NB15", "IoT23", "MQTTset", "TON_IoT", "Edge_IIoTset"] if d in set(F.dataset) & set(Cc.dataset)]
    f = per_ds(F).loc[ds_common]; c = per_ds(Cc).loc[ds_common]
    methods = [m for m in ORDER if m in f.columns and m in c.columns]
    log = []
    P = lambda *x: (print(*x), log.append(" ".join(str(y) for y in x)))
    P("=" * 70); P("FILE-ORDER vs CHRONOLOGICAL  (mean CL-F1 over seeds)"); P("=" * 70)
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Robustness of the stream construction: CL-F1 of every arm when the same subsample is streamed in released-file order (F) and in timestamp order (T), on the datasets whose files carry a timestamp. Rank columns give each arm's rank within the stream (1 = best).}",
         r"\label{tab:hybrid_chrono}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{l" + "cccc" * len(ds_common) + "}", r"\toprule",
         "Arm & " + " & ".join(r"\multicolumn{4}{c}{%s}" % NICE[d] for d in ds_common) + r" \\",
         " ".join(r"\cmidrule(lr){%d-%d}" % (2 + 4 * i, 5 + 4 * i) for i in range(len(ds_common))),
         " & " + " & ".join("F & T & rk$_F$ & rk$_T$" for _ in ds_common) + r" \\", r"\midrule"]
    rf = f[methods].rank(axis=1, ascending=False); rc = c[methods].rank(axis=1, ascending=False)
    for m in methods:
        cells = []
        for d in ds_common:
            cells += ["%.4f" % f.loc[d, m], "%.4f" % c.loc[d, m], "%d" % rf.loc[d, m], "%d" % rc.loc[d, m]]
            P("  %-18s %-12s file=%.4f  time=%.4f  (rank %d -> %d)" % (m, d, f.loc[d, m], c.loc[d, m], rf.loc[d, m], rc.loc[d, m]))
        L.append(m + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    open(os.path.join(a.outdir, "tab_hybrid_chrono.tex"), "w").write("\n".join(L))
    # rank agreement per dataset
    P("\nSpearman rank correlation of arms, file-order vs chronological:")
    for d in ds_common:
        r = stats.spearmanr(f.loc[d, methods], c.loc[d, methods]).correlation
        P("  %-12s rho=%.3f   best(F)=%s  best(T)=%s" % (d, r, f.loc[d, methods].idxmax(), c.loc[d, methods].idxmax()))
    # expert ordering per dataset (which expert leads) under both orders
    for name, D in (("file", F), ("time", Cc)):
        h = D[D.method == "CADENCE-H"].groupby("dataset")[["f1_B", "f1_O", "f1", "w"]].mean()
        P("\n%s-order experts: " % name); P(h.loc[[d for d in ds_common if d in h.index]].round(4).to_string())
    open(os.path.join(a.outdir, "chrono_summary.txt"), "w").write("\n".join(log)); print("[written]", a.outdir)

if __name__ == "__main__":
    main()
