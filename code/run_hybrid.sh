#!/usr/bin/env bash
# =============================================================================
# CADENCE-H  — full experiment, tables and figures   (LINUX server)
#
#   cd <repo>/code
#   bash run_hybrid.sh              # smoke test, then asks before the full run
#   bash run_hybrid.sh --full       # full run only (use with nohup)
#   bash run_hybrid.sh --report     # tables + figures from an existing CSV
#
# Long run:
#   nohup bash run_hybrid.sh --full > hybrid_run.log 2>&1 &
#   tail -f hybrid_run.log
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python3}
DATA=${DATA:-./data}
SEEDS=${SEEDS:-"42 123 456"}
OUT=${OUT:-hybrid_results.csv}
SUB=${SUBSTAGES:-1}          # set SUBSTAGES=2 for finer adaptation granularity
BETA=${BETA:-30}
GATE=${GATE:-prequential}   # or: guard

report() {
    echo "=== tables ==="
    $PY make_hybrid_tables.py  --results "$OUT" --outdir tables_hybrid
    echo "=== figures ==="
    $PY make_hybrid_figures.py --results "$OUT" --outdir figs_hybrid
    echo
    echo "Send back:  $OUT   tables_hybrid/   figs_hybrid/   and the nohup log"
}

if [[ "${1:-}" == "--report" ]]; then report; exit 0; fi

# ---- rerun 2 (one job): chronological CIC-IDS2017 x 12 arms  +  SRP online-expert arms x 6 datasets ----
#   prerequisite for part (a): data/CICIDS2017_chrono/cicids2017_chrono.csv built by prepare_cicids_raw.py
#   bash run_hybrid.sh --rerun2
if [[ "${1:-}" == "--rerun2" ]]; then
    if [[ -f data/CICIDS2017_chrono/cicids2017_chrono.csv ]]; then
        echo "=== RERUN 2a: CICIDS2017_chrono (timestamp order) x 3 seeds x 12 arms ==="
        $PY cadence_hybrid.py --data-root "$DATA" --datasets CICIDS2017_chrono --seeds $SEEDS --substages "$SUB" --beta "$BETA" --gate "$GATE" --order time --out hybrid_results_chrono.csv
    else
        echo "=== RERUN 2a skipped: data/CICIDS2017_chrono/cicids2017_chrono.csv not found (run prepare_cicids_raw.py first) ==="
    fi
    echo "=== RERUN 2b: SRP online-expert arms (SRP, CADENCE-H-srp, DWM-srp) x 6 datasets x 3 seeds ==="
    $PY cadence_hybrid.py --data-root "$DATA" --seeds $SEEDS --substages "$SUB" --beta "$BETA" --gate "$GATE" --methods SRP CADENCE-H-srp DWM-srp --out hybrid_results_srp.csv
    echo "=== rerun-2 tables ==="
    $PY make_rerun2_tables.py --file-order "$OUT" --chrono hybrid_results_chrono.csv --srp hybrid_results_srp.csv --outdir tables_hybrid
    echo
    echo "Send back:  hybrid_results_chrono.csv  hybrid_results_srp.csv  tables_hybrid/tab_hybrid_chrono.tex  tables_hybrid/tab_hybrid_srp.tex  tables_hybrid/rerun2_summary.txt  and the nohup log"
    exit 0
fi

# ---- (older) chronological-only mode; kept for reference ----
#   bash run_hybrid.sh --chrono            # CICIDS2017 UNSW_NB15 IoT23 by default
#   CHRONO_DS="CICIDS2017" bash run_hybrid.sh --chrono
if [[ "${1:-}" == "--chrono" ]]; then
    CHRONO_DS=${CHRONO_DS:-"CICIDS2017 UNSW_NB15 IoT23"}
    COUT=${COUT:-hybrid_results_chrono.csv}
    echo "=== CHRONOLOGICAL RUN: $CHRONO_DS x 3 seeds x 12 methods  (--order time) ==="
    $PY cadence_hybrid.py --data-root "$DATA" --datasets $CHRONO_DS --seeds $SEEDS --substages "$SUB" --beta "$BETA" --gate "$GATE" --order time --out "$COUT"
    echo "=== file-order vs chronological comparison ==="
    $PY make_chrono_table.py --file-order "$OUT" --chrono "$COUT" --outdir tables_hybrid
    echo
    echo "Send back:  $COUT   tables_hybrid/tab_hybrid_chrono.tex   tables_hybrid/chrono_summary.txt   and the nohup log"
    exit 0
fi

# ---- preflight ---------------------------------------------------------------
echo "workdir: $(pwd)   python: $($PY --version 2>&1)   data: $DATA"
miss=0
for f in cadence_hybrid.py make_hybrid_tables.py make_hybrid_figures.py \
         cadence_core.py cadence_runner.py cadence_temporal.py; do
    [[ -f "$f" ]] && echo "  found   $f" || { echo "  MISSING $f"; miss=1; }
done
[[ -d "$DATA" ]] && echo "  found   $DATA/" || { echo "  MISSING $DATA/"; miss=1; }
[[ $miss -ne 0 ]] && { echo "Preflight failed." >&2; exit 1; }
$PY - <<'PYCHK'
import importlib.util, sys
need=["numpy","pandas","sklearn","xgboost","scipy","imblearn","river","matplotlib"]
opt=[m for m in ["statsmodels"] if importlib.util.find_spec(m) is None]
bad=[m for m in need if importlib.util.find_spec(m) is None]
print("  python deps:", "OK" if not bad else "MISSING "+", ".join(bad))
if bad: print("  -> pip3 install --user " + " ".join({"sklearn":"scikit-learn","imblearn":"imbalanced-learn"}.get(m,m) for m in bad))
if opt: print("  optional (stage-level mixed model): pip3 install --user statsmodels")
sys.exit(1 if bad else 0)
PYCHK
echo

if [[ "${1:-}" == "--full" ]]; then
    echo "=== FULL RUN v4: 6 datasets x 3 seeds x 12 methods  (gate=$GATE, beta=$BETA, substages=$SUB) ==="
    $PY cadence_hybrid.py --data-root "$DATA" --seeds $SEEDS --substages "$SUB" --beta "$BETA" --gate "$GATE" --out "$OUT"
    report
    exit 0
fi

# ---- smoke test --------------------------------------------------------------
echo "=== SMOKE TEST: CICIDS2017 + IoT23, seed 42, budget 8000, 8 methods ==="
$PY cadence_hybrid.py --data-root "$DATA" --datasets CICIDS2017 IoT23 --seeds 42 --budget 8000 \
    --methods Static-XGB CADENCE-R CADENCE-R-leaky ARF PWPAE-fair PWPAE-interleaved CADENCE-H DWM --substages "$SUB" --beta "$BETA" --gate "$GATE" --out hybrid_smoke.csv
echo
echo "Smoke test done -> hybrid_smoke.csv"
echo "What to look for on CICIDS2017:"
echo "  * F_O (ARF) alone should beat F_B (CADENCE-R) alone   (online wins on class emergence)"
echo "  * mean w should be LOW there (gate trusting the online expert) and HIGH on IoT23"
echo "  * CADENCE-H should be >= the better expert on each stream"
echo "  * PWPAE-interleaved should exceed PWPAE-fair (earlier labels); CADENCE-R-leaky should accept more than CADENCE-R"
echo "  * bestfix=... shows the best fixed weight from the sweep; compare with CADENCE-H"
echo
read -r -p "Press Enter for the FULL run, or Ctrl-C to stop (then use nohup ... --full) " _
$PY cadence_hybrid.py --data-root "$DATA" --seeds $SEEDS --substages "$SUB" --beta "$BETA" --gate "$GATE" --out "$OUT"
report
