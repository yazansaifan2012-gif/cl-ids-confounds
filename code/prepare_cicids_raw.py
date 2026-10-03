#!/usr/bin/env python3
# =============================================================================
# Build a CHRONOLOGICAL CIC-IDS2017 file from the raw per-day flow CSVs.
#
#   Needs the "GeneratedLabelledFlows" release (TrafficLabelling folder), whose
#   CSVs carry Flow ID, Source IP, ..., Timestamp, ..., Label.  (The
#   MachineLearningCSV release has NO timestamp column and cannot be used.)
#
#   python3 prepare_cicids_raw.py --raw-dir data/CICIDS2017_raw \
#       --out data/CICIDS2017_chrono/cicids2017_chrono.csv \
#       --reference data/CICIDS2017/cicids2017_cleaned.csv
#
# What it does
#   1. reads the eight day files in capture order (Mon -> Fri), latin-1 encoding
#   2. strips whitespace from column names, drops rows with no label
#   3. maps the 15 fine labels to the 7 "Attack Type" groups used by the cleaned
#      file, so the chronological stream has the same label space as the file-order
#      stream; unmapped labels are dropped and reported
#   4. keeps Timestamp (used by --order time, dropped from features by the loader)
#   5. writes one CSV and prints the class counts of both files side by side
# =============================================================================
import argparse, glob, os, re, sys
import pandas as pd

DAY_ORDER = ["monday", "tuesday", "wednesday", "thursday-workinghours-morning", "thursday-workinghours-afternoon",
             "friday-workinghours-morning", "friday-workinghours-afternoon-portscan", "friday-workinghours-afternoon-ddos"]

# fine label (lower-cased, dashes/spaces normalised) -> Attack Type group of the cleaned file
GROUP = {
    "benign": "Benign",
    "dos hulk": "DoS", "dos goldeneye": "DoS", "dos slowloris": "DoS", "dos slowhttptest": "DoS",
    "ddos": "DDoS",
    "portscan": "Port Scan",
    "ftp-patator": "Brute Force", "ssh-patator": "Brute Force",
    "web attack brute force": "Web Attack", "web attack xss": "Web Attack", "web attack sql injection": "Web Attack",
    "bot": "Bot",
    # Infiltration and Heartbleed (36 and 11 flows) are not in the 7-class cleaned file; dropped
}

def norm(lbl):
    s = str(lbl).strip().lower()
    s = s.replace("\x96", "-").replace("–", "-").replace("—", "-")   # latin-1 / unicode dashes
    s = re.sub(r"\s*-\s*", " ", s) if s.startswith("web attack") else s        # "web attack – xss" -> "web attack xss"
    return re.sub(r"\s+", " ", s)

def day_key(path):
    b = os.path.basename(path).lower()
    for i, k in enumerate(DAY_ORDER):
        if k in b.replace("_", "-"):
            return i
    return 99

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--reference", default=None, help="the cleaned file, to print its class counts for comparison")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.raw_dir, "*.csv")), key=day_key)
    if len(files) < 8:
        print("WARNING: expected 8 day files, found %d:" % len(files)); [print("   ", f) for f in files]
    frames = []; dropped = {}
    for f in files:
        df = pd.read_csv(f, encoding="latin-1", low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        if "Label" not in df.columns or "Timestamp" not in df.columns:
            sys.exit("ERROR: %s has no Label/Timestamp column — this is not the GeneratedLabelledFlows release" % f)
        df = df[df["Label"].notna()]
        g = df["Label"].map(lambda x: GROUP.get(norm(x)))
        for lbl, n in df.loc[g.isna(), "Label"].value_counts().items():
            dropped[lbl] = dropped.get(lbl, 0) + int(n)
        df = df[g.notna()].copy(); df["Attack Type"] = g[g.notna()].values
        df = df.drop(columns=["Label"])
        print("  %-62s %8d rows kept" % (os.path.basename(f), len(df)), flush=True)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True); out.to_csv(a.out, index=False)
    print("\n[written] %s  (%d rows, %d columns)" % (a.out, len(out), out.shape[1]))
    if dropped:
        print("dropped labels (not in the 7-class space):", dropped)
    print("\nclass counts, chronological file:"); print(out["Attack Type"].value_counts().to_string())
    if a.reference and os.path.exists(a.reference):
        ref = pd.read_csv(a.reference, usecols=["Attack Type"], low_memory=False)
        print("\nclass counts, cleaned (file-order) file:"); print(ref["Attack Type"].value_counts().to_string())
        missing = set(ref["Attack Type"].unique()) ^ set(out["Attack Type"].unique())
        print("\nlabel-space mismatch:", missing if missing else "none — same 7 classes")

if __name__ == "__main__":
    main()
