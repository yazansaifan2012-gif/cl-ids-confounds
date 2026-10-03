# Quantifying Evaluation Confounds in Continual Intrusion Detection

Code, per-stage results and run logs for the paper *"Quantifying Evaluation Confounds in Continual Intrusion Detection: Label-Release Timing, Leakage-Free Admission, and Expert Gating"* (under review). Every table, figure and statistic in the paper can be regenerated from the files in `results/` without re-running the experiments.

## What is here

```
code/
  cadence_hybrid.py       all 15 arms; one prequential, stage-level protocol for every arm
  cadence_core.py         XGBoost dual-ensemble batch backbone, calibration, SMOTE, focal weights
  cadence_runner.py       dataset loading, path mapping, balanced class-floor subsampling
  cadence_temporal.py     order-preserving stream construction (file order, or timestamp order)
  make_hybrid_tables.py   all LaTeX tables + Friedman / ranks / Holm / stage-level mixed model
  make_hybrid_figures.py  all result figures
  make_rerun2_tables.py   second-online-backbone (SRP) table; chronological comparison table
  make_chrono_table.py    file-order vs timestamp-order comparison
  prepare_cicids_raw.py   builds a timestamp-ordered CIC-IDS2017 file from the raw per-day flows
  run_hybrid.sh           runner: smoke test, --full, --rerun2, --report
results/
  results_main_12arms.csv   1,728 rows: 12 arms x 6 datasets x 3 seeds x 8 stages
  results_srp_backbone.csv  432 rows: SRP online-expert arms
  tables/, figures/         exactly as generated for the paper
logs/                       run logs of both experiments
```

## Arms

| Arm | What it is |
|---|---|
| Static-XGB | initialization model, never updated |
| Naive-XGB | batch-only candidate, always admitted |
| CADENCE-B | batch-only candidate, admitted on a disjoint guard |
| ER-XGB | candidate with exemplar rehearsal, always admitted |
| CADENCE-R | rehearsal candidate admitted on a disjoint guard |
| CADENCE-R-leaky | **protocol arm**: as CADENCE-R, guard drawn from the candidate's own training records |
| ARF / SRP | online expert alone |
| PWPAE-fair | PWPAE ensemble, whole stage scored before any member learns |
| PWPAE-interleaved | **protocol arm**: same members, learn each record right after predicting it |
| DWM, DWM-srp | Dynamic Weighted Majority over the batch and online experts |
| CADENCE-H, CADENCE-H-srp | prequentially weighted batch-online mixture |
| CADENCE-H-ng | mixture with admission disabled |

Fixed-weight (w = 0.0 ... 1.0), calibration-weight and hard-selection rules are computed from the saved expert scores of the CADENCE-H run (columns `f1_fix00` ... `f1_fix10`, `f1_fixcal`, `f1_hard`).

## Reproduce the paper's tables and figures (no data needed)

```bash
pip install -r requirements.txt
cd code
python3 make_hybrid_tables.py  --results ../results/results_main_12arms.csv --outdir ../out/tables
python3 make_hybrid_figures.py --results ../results/results_main_12arms.csv --outdir ../out/figures
python3 make_rerun2_tables.py  --file-order ../results/results_main_12arms.csv --chrono none --srp ../results/results_srp_backbone.csv --outdir ../out/tables
```

## Re-run the experiments

Download the six public datasets from their providers and place one CSV per dataset in `data/<name>/`
(`TON_IoT`, `CICIDS2017`, `Edge_IIoTset`, `UNSW_NB15`, `IoT23`, `MQTTset`). Then:

```bash
cd code
bash run_hybrid.sh                 # smoke test, then asks
bash run_hybrid.sh --full          # 12 arms x 6 datasets x 3 seeds (~2 h on one CPU node)
bash run_hybrid.sh --rerun2        # SRP backbone (+ chronological CICIDS2017 if prepared)
```

Configuration (fixed for every dataset and arm): batch expert XGBoost dual ensemble (depth 7 / 5, eta 0.05 / 0.08, 160 rounds, early stopping on the calibration split), borderline-SMOTE 0.55, focal gamma 1.5, rehearsal buffer 200 per class; online expert ARF with 10 trees; gate beta = 30, alpha = 0.7; guard fraction 25%; seeds 42, 123, 456.

## Chronological CIC-IDS2017 (optional)

The preprocessed CSVs used in the paper carry no timestamps. To build a capture-ordered stream, download *GeneratedLabelledFlows.zip* from the CIC-IDS2017 page, put the eight `*.pcap_ISCX.csv` files in `data/CICIDS2017_raw/`, and run

```bash
python3 prepare_cicids_raw.py --raw-dir data/CICIDS2017_raw --out data/CICIDS2017_chrono/cicids2017_chrono.csv
```

## Environment

Python 3.11.9, XGBoost 3.2.0, scikit-learn 1.8.0, imbalanced-learn 0.14.2, river 0.25.0, NumPy 2.4.4, SciPy 1.17.1, statsmodels 0.15.0.

## License

MIT (see `LICENSE`). Datasets remain under their providers' terms and are not redistributed here.
