#!/usr/bin/env python3
# CADENCE - FULL BALANCED BENCHMARK RUNNER (run on your server)
# Re-runs methods on BALANCED class-floor sampling so long-tailed attack classes
# get fair representation. Leakage guards unchanged.
# Methods: Static XGBoost, Fixed Replay (ER), PSI-only, CADENCE
#          (+ EWC, LwF with --include-torch)
#
# QUICK START:
#   py cadence_runner.py --data-root ./data --datasets CICIDS2017 UNSW_NB15 IoT23 --include-torch
#   py make_table.py --results cadence_balanced_results.csv --out Table_IV_updated
#
# --data-root auto-detects the .csv inside each dataset folder (no DS_CONFIGS edits).
# OUTPUT: cadence_balanced_results.csv
from __future__ import annotations
import argparse, glob, importlib.util, os, sys, time, gc, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")


def load_core(path):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        sys.exit("[!] cadence_core not found at %s (use --core)." % path)
    spec = importlib.util.spec_from_file_location("cadence_core", path)
    mod = importlib.util.module_from_spec(spec); sys.modules["cadence_core"] = mod
    spec.loader.exec_module(mod); return mod


def remap_paths(C, data_root):
    if not data_root: return
    data_root = os.path.abspath(data_root)
    aliases = {"CICIDS2017": ["CICIDS2017", "CICIDS", "cicids2017"],
               "UNSW_NB15": ["UNSW_NB15", "UNSW-NB15", "unsw_nb15"],
               "IoT23": ["IoT23", "IoT-23", "iot23"],
               "TON_IoT": ["TON_IoT", "TON-IoT", "ton_iot"],
               "Edge_IIoTset": ["Edge_IIoTset", "Edge-IIoTset", "edge_iiotset"],
               "MQTTset": ["MQTTset", "mqttset"]}
    for key, cfg in C.DS_CONFIGS.items():
        for folder in aliases.get(key, [key]):
            d = os.path.join(data_root, folder)
            if os.path.isdir(d):
                csvs = sorted(glob.glob(os.path.join(d, "*.csv")))
                if csvs:
                    cfg["path"] = csvs[0]; cfg["_abs"] = True
                    print("  [data] %-14s -> %s" % (key, csvs[0]), flush=True); break


def _csv_path(C, cfg):
    return cfg["path"] if cfg.get("_abs") else os.path.join(C.ROOT, cfg["path"])


def _waterfill(counts, budget):
    counts = {k: int(v) for k, v in counts.items()}
    lo, hi = 1, max(counts.values())
    for _ in range(60):
        mid = (lo + hi) // 2
        if sum(min(c, mid) for c in counts.values()) < budget: lo = mid + 1
        else: hi = mid
    return {k: min(c, lo) for k, c in counts.items()}


def load_balanced(C, ds_key, seed, lowmem=False, budget=None):
    cfg = C.DS_CONFIGS[ds_key]; path = _csv_path(C, cfg)
    label_col = cfg["label_col"]; budget = budget or cfg["n_samples"]
    np.random.seed(seed); rng = np.random.default_rng(seed)

    if lowmem:
        counts = {}
        for ch in pd.read_csv(path, usecols=[label_col], chunksize=400000, low_memory=False):
            for k, v in ch[label_col].astype(str).value_counts().items():
                counts[k] = counts.get(k, 0) + int(v)
    else:
        df_full = pd.read_csv(path, low_memory=False)
        counts = {str(k): int(v) for k, v in df_full[label_col].astype(str).value_counts().items()}
    take = _waterfill(counts, budget)
    print("    [%s] %d classes | budget %d | targets %s"
          % (ds_key, len(counts), budget, {k: take[k] for k in sorted(take, key=lambda x: -take[x])}), flush=True)

    if lowmem:
        res_df, res_key = {}, {}
        for ch in pd.read_csv(path, chunksize=400000, low_memory=False):
            ch["__l"] = ch[label_col].astype(str)
            for cls, sub in ch.groupby("__l", sort=False):
                k = take.get(str(cls), 2); keys = rng.random(len(sub))
                if cls in res_df:
                    sub = pd.concat([res_df[cls], sub], ignore_index=True)
                    keys = np.concatenate([res_key[cls], keys])
                if len(sub) > k:
                    top = np.argpartition(-keys, k)[:k]
                    sub = sub.iloc[top].reset_index(drop=True); keys = keys[top]
                res_df[cls] = sub; res_key[cls] = keys
            del ch; gc.collect()
        df = pd.concat(list(res_df.values()), ignore_index=True).drop(columns="__l"); del res_df, res_key
    else:
        parts = [g.sample(n=min(take.get(str(cls), 2), len(g)), random_state=seed)
                 for cls, g in df_full.groupby(df_full[label_col].astype(str), sort=False)]
        df = pd.concat(parts, ignore_index=True); del df_full
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True); gc.collect()

    le = C.LabelEncoder(); y_all = le.fit_transform(df[label_col].astype(str))
    drop = [c for c in cfg["drop_cols"] + [label_col] if c in df.columns]
    df = df.drop(columns=drop)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = C.LabelEncoder().fit_transform(df[col].astype(str).fillna("NA"))
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    X = np.nan_to_num(df.values.astype(np.float32)); del df; gc.collect()
    NC = int(len(le.classes_))
    print("    %d samples | %d classes | %d features" % (len(y_all), NC, X.shape[1]), flush=True)
    X, y_all = C._deduplicate(X, y_all)
    X_dev, X_te, y_dev, y_te = C.train_test_split(X, y_all, test_size=0.20, random_state=seed, stratify=y_all); del X; gc.collect()
    X_init, X_stream, y_init, y_stream = C.train_test_split(X_dev, y_dev, test_size=0.35, random_state=seed, stratify=y_dev); del X_dev; gc.collect()
    nzv = X_init.var(axis=0) > 1e-6
    X_init, X_stream, X_te = X_init[:, nzv], X_stream[:, nzv], X_te[:, nzv]
    sc = C.RobustScaler(quantile_range=(5, 95)).fit(X_init)
    X_init = np.nan_to_num(sc.transform(X_init).astype(np.float32))
    X_stream = np.nan_to_num(sc.transform(X_stream).astype(np.float32))
    X_te = np.nan_to_num(sc.transform(X_te).astype(np.float32))
    try:
        X_tr, X_val, y_tr, y_val = C.train_test_split(X_init, y_init, test_size=0.15, random_state=seed, stratify=y_init)
    except ValueError:
        X_tr, X_val, y_tr, y_val = C.train_test_split(X_init, y_init, test_size=0.15, random_state=seed)
    del X_init; gc.collect()
    bi = np.array_split(np.random.default_rng(seed).permutation(len(y_stream)), C.N_BATCHES)
    return X_tr, X_val, X_stream, X_te, y_tr, y_val, y_stream, y_te, bi, NC, le


def run_cl_tuned(C, m_main, m_div, wt, wd, thr, X_tr, y_tr, X_val, y_val, X_stream, y_stream, X_te, y_te,
                 nc, seed, use_context=True, guarded=True, rehearsal=True, max_per_class=900, age_decay=0.02,
                 rounds_full=160, rounds_partial=120):
    """Guarded CADENCE continual adaptation.
    rehearsal=True: retrain the dual-ensemble on the ACCUMULATED labeled stream
    (X_tr + stream-so-far) with the strong static config -- the model genuinely
    uses the stream data that the frozen baseline ignores. guarded=True: an update
    is accepted only if it does not reduce held-out validation macro-F1 (no
    negative transfer)."""
    import xgboost as xgb
    from sklearn.metrics import f1_score
    def _valf1(mm_, md_, wt_, wd_, thr_):
        yp_ = C.apply_thresholds(C._ensemble_probs(mm_, md_, X_val, nc, wt_, wd_), thr_)
        return f1_score(y_val, yp_, average='macro', zero_division=0)
    def _train_dual(X, y, rounds):
        Xa, ya = C.smote_safe(X, y, nc, ratio=0.55, k=3, severe=False, seed=seed)
        bw = C.eff_num_weights(ya)
        mm = C._train_xgb(Xa, ya, X_val, y_val, nc, bw, "main", rounds)
        p = mm.predict(xgb.DMatrix(Xa)).reshape(-1, nc).astype(np.float32)
        md = C._train_xgb(Xa, ya, X_val, y_val, nc, C.focal_proxy_weights(ya, p, gamma=1.5, base_w=bw), "div", rounds)
        wt2, wd2 = C._val_weights(mm, md, X_val, y_val, nc)
        thr2 = C.calibrate_thresholds(C._ensemble_probs(mm, md, X_val, nc, wt2, wd2), y_val, nc, 45)
        return mm, md, wt2, wd2, thr2
    drift = C.AdaptiveDriftDetector(psi_gradual=0.10, psi_sudden=0.25, psi_ood=0.45,
                                    conf_ood_gate=0.40, history_len=6, recurring_thr=0.92, hysteresis=1)
    ctx = C.ContextStateEngine(X_tr, y_tr, nc) if use_context else None
    pol_eng = C.ContextGatedPolicy() if use_context else None
    Xacc, yacc = X_tr.copy(), y_tr.copy()
    batches = np.array_split(np.random.default_rng(seed).permutation(len(y_stream)), C.N_BATCHES)
    out = []; last = -99
    for bi, bidx in enumerate(batches):
        Xb, yb = X_stream[bidx], y_stream[bidx]
        p_b = C._ensemble_probs(m_main, m_div, Xb, nc, wt, wd); mc = float(p_b.max(1).mean())
        di = drift.update(Xb, mean_conf=mc); pst = di["state"]; pint = drift.retrain_intensity()
        if use_context:
            cs, csc, cst = ctx.analyse(Xb, yb, di["severity"])
            pol = pol_eng.get(cs, pst, pint, csc, cst["ctx_score_trend"], cst["n_classes_seen"],
                              cst["class_entropy"], cst["imbalance"]); inten = pol.intensity
        else:
            pol = C.AdaptationPolicy(intensity=pint, note="no_ctx"); inten = pint
        Xacc = np.vstack([Xacc, Xb]); yacc = np.concatenate([yacc, yb])   # accumulate labeled stream
        if (bi - last) < 1 and inten != "full": inten = "skip"
        if inten != "skip":
            rnd = rounds_full if inten == "full" else rounds_partial
            c_main, c_div, c_wt, c_wd, c_thr = _train_dual(Xacc, yacc, rnd)
            if (not guarded) or (_valf1(c_main, c_div, c_wt, c_wd, c_thr) >= _valf1(m_main, m_div, wt, wd, thr)):
                m_main, m_div, wt, wd, thr = c_main, c_div, c_wt, c_wd, c_thr
            last = bi
        eff = (thr * pol.thr_mult).astype(np.float32)
        yp = C.apply_thresholds(C._ensemble_probs(m_main, m_div, X_te, nc, wt, wd), eff)
        out.append(dict(f1_macro=f1_score(y_te, yp, average="macro", zero_division=0), fpr_macro=C.fpr_macro(y_te, yp, nc)))
        del p_b; gc.collect()
    return out


def run_er(C, m_main, m_div, wt, wd, thr, X_tr, y_tr, X_stream, y_stream, X_te, y_te, nc, seed, rounds=90):
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(seed); bX = [X_tr]; bY = [y_tr]
    def samp(n):
        Xa = np.vstack(bX); Ya = np.concatenate(bY)
        if len(Ya) <= n: return Xa, Ya
        i = rng.choice(len(Ya), n, replace=False); return Xa[i], Ya[i]
    out = []
    for bidx in np.array_split(rng.permutation(len(y_stream)), C.N_BATCHES):
        Xb, yb = X_stream[bidx], y_stream[bidx]
        RX, RY = samp(max(len(Xb) * 6, 4000)); Xc = np.vstack([Xb, RX]); yc = np.concatenate([yb, RY])
        mn = np.bincount(yc, minlength=nc).min()
        try: Xt, Xv, yt, yv = train_test_split(Xc, yc, test_size=0.15, random_state=seed, stratify=yc if mn >= 2 else None)
        except Exception: Xt, Xv, yt, yv = train_test_split(Xc, yc, test_size=0.15, random_state=seed)
        Xa, ya = C.smote_safe(Xt, yt, nc, ratio=0.55, k=3, severe=False, seed=seed); bw = C.eff_num_weights(ya)
        m_main = C._train_xgb(Xa, ya, Xv, yv, nc, bw, "main", rounds)
        p = m_main.predict(xgb.DMatrix(Xa)).reshape(-1, nc).astype(np.float32)
        m_div = C._train_xgb(Xa, ya, Xv, yv, nc, C.focal_proxy_weights(ya, p, gamma=1.5, base_w=bw), "div", rounds)
        wt, wd = C._val_weights(m_main, m_div, Xv, yv, nc)
        yp = C._ensemble_probs(m_main, m_div, X_te, nc, wt, wd).argmax(1)
        out.append(dict(f1_macro=f1_score(y_te, yp, average="macro", zero_division=0), fpr_macro=C.fpr_macro(y_te, yp, nc)))
        bX.append(Xb); bY.append(yb); gc.collect()
    return out


def run_der(C, X_tr, y_tr, X_val, y_val, X_stream, y_stream, X_te, y_te, nc, seed, alpha=0.5, buf_size=500):
    """Dark Experience Replay (DER, Buzzega et al. 2020): replay stored soft logits
    with an MSE distillation term while learning each new stream batch on the MLP
    backbone. Buffer holds (x, logits) via reservoir sampling."""
    import torch, torch.nn as nn, torch.nn.functional as F
    torch.manual_seed(seed); np.random.seed(seed)
    model = safe_train_initial_mlp(C, X_tr, y_tr, X_val, y_val, nc, seed)
    rng = np.random.default_rng(seed)
    # reservoir buffer of (x, logits) seeded from init training data
    model.eval()
    with torch.no_grad():
        z0 = model(torch.FloatTensor(X_tr)).numpy()
    idx = rng.choice(len(X_tr), min(buf_size, len(X_tr)), replace=False)
    buf_x = list(X_tr[idx]); buf_z = list(z0[idx]); seen = len(X_tr)
    def reservoir_add(xs, zs):
        nonlocal seen, buf_x, buf_z
        for x, z in zip(xs, zs):
            seen += 1
            if len(buf_x) < buf_size:
                buf_x.append(x); buf_z.append(z)
            else:
                j = rng.integers(0, seen)
                if j < buf_size: buf_x[j] = x; buf_z[j] = z
    out = []
    for bidx in np.array_split(rng.permutation(len(y_stream)), C.N_BATCHES):
        Xb, yb = X_stream[bidx], y_stream[bidx]
        opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
        ce = nn.CrossEntropyLoss()
        xt = torch.FloatTensor(Xb); yt = torch.LongTensor(yb)
        bx = torch.FloatTensor(np.array(buf_x)); bz = torch.FloatTensor(np.array(buf_z))
        model.train()
        for _ in range(30):
            opt.zero_grad()
            loss = ce(model(xt), yt)
            if len(buf_x) > 0:                      # DER distillation on stored logits
                k = min(len(buf_x), max(64, len(Xb)))
                sel = rng.choice(len(buf_x), k, replace=False)
                loss = loss + alpha * F.mse_loss(model(bx[sel]), bz[sel])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        model.eval()
        with torch.no_grad():
            zb = model(xt).numpy()
        reservoir_add(Xb, zb)
        f1, fpr = C.eval_mlp(model, X_te, y_te, nc); out.append(dict(f1_macro=f1, fpr_macro=fpr))
    return out


def safe_train_initial_mlp(C, X_tr, y_tr, X_val, y_val, nc, seed):
    """Collapse-guarded initial MLP training. The default lr can diverge to a
    one-class solution on some datasets (e.g. Edge-IIoTset -> F1~0.04). Retry at
    lower learning rates and keep the best-validating model."""
    orig = C.MLP_LR_INIT; best, best_f1 = None, -1.0
    for lr in (orig, 3e-4, 1e-4):
        C.MLP_LR_INIT = lr
        m = C.train_initial_mlp(X_tr, y_tr, X_val, y_val, nc, seed)
        f1, _ = C.eval_mlp(m, X_val, y_val, nc)
        if f1 > best_f1: best_f1, best = f1, m
        if f1 >= 0.30: break          # healthy model -> stop retrying
    C.MLP_LR_INIT = orig
    if best_f1 < 0.30:
        print("    [warn] MLP still weak (val F1=%.3f) after lr retries" % best_f1, flush=True)
    return best


def run_ewc_lwf(C, method, X_tr, y_tr, X_val, y_val, X_stream, y_stream, X_te, y_te, nc, seed):
    import torch, torch.nn.functional as F, copy
    torch.manual_seed(seed); np.random.seed(seed)
    model = safe_train_initial_mlp(C, X_tr, y_tr, X_val, y_val, nc, seed)
    if method == "EWC":
        fisher = C.compute_fisher(model, X_tr, y_tr, nc)
        snap = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    else:
        teacher = copy.deepcopy(model)
    ref = {"r": None}
    def psi_update(Xb, mc):
        if ref["r"] is None: ref["r"] = Xb.copy(); return "stable"
        psi = C._feature_psi(ref["r"], Xb)
        s = "ood" if (psi >= 0.45 and mc < 0.40) else "sudden_drift" if psi >= 0.25 else "gradual_drift" if psi >= 0.10 else "stable"
        if psi < 0.10:
            n = min(len(ref["r"]), len(Xb)); ref["r"] = 0.8 * ref["r"][:n] + 0.2 * Xb[:n]
        elif s == "sudden_drift": ref["r"] = Xb.copy()
        return s
    def intensity(s): return "full" if s in ("sudden_drift", "ood") else "partial" if s == "gradual_drift" else "skip"
    out = []; last = -99
    for b, bidx in enumerate(np.array_split(np.random.default_rng(seed).permutation(len(y_stream)), C.N_BATCHES)):
        Xb, yb = X_stream[bidx], y_stream[bidx]
        model.eval()
        with torch.no_grad():
            mc = float(F.softmax(model(torch.FloatTensor(Xb)), 1).max(1).values.mean())
        st = psi_update(Xb, mc); inten = intensity(st)
        if (b - last) < 1 and inten != "full": inten = "skip"
        if inten != "skip":
            if method == "EWC":
                model = C.train_ewc_mlp(model, fisher, snap, Xb, yb, X_val, y_val, nc, seed)
                nf = C.compute_fisher(model, Xb, yb, nc)
                for n in fisher: fisher[n] = 0.5 * fisher[n] + 0.5 * nf[n]
                snap = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
            else:
                model = C.train_lwf_mlp(model, teacher, Xb, yb, X_val, y_val, nc, seed); teacher = copy.deepcopy(model)
            last = b
        f1, fpr = C.eval_mlp(model, X_te, y_te, nc); out.append(dict(f1_macro=f1, fpr_macro=fpr))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", default="cadence_core.py")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--datasets", nargs="+", default=["CICIDS2017", "UNSW_NB15", "IoT23"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    ap.add_argument("--methods", nargs="+",
                    default=["Static XGBoost", "Fixed Replay (ER)", "PSI-only", "CADENCE"])
    ap.add_argument("--include-torch", action="store_true")
    ap.add_argument("--no-guard", action="store_true", help="disable validation-gated safe adaptation")
    ap.add_argument("--lowmem", action="store_true")
    ap.add_argument("--out", default="cadence_balanced_results.csv")
    args = ap.parse_args()

    C = load_core(args.core); remap_paths(C, args.data_root)
    methods = list(args.methods)
    if args.include_torch:
        if getattr(C, "_TORCH_OK", False): methods += ["EWC", "LwF", "DER"]
        else: print("[!] --include-torch but torch not available; skipping EWC/LwF.")

    rows = []
    for ds in args.datasets:
        if ds not in C.DS_CONFIGS:
            print("[!] skip unknown dataset %s" % ds); continue
        lowmem = args.lowmem or ds == "CICIDS2017"
        for seed in args.seeds:
            print("\n" + "=" * 64 + "\n  %s  seed=%d\n" % (ds, seed) + "=" * 64, flush=True)
            t0 = time.time()
            X_tr, X_val, X_stream, X_te, y_tr, y_val, y_stream, y_te, _, NC, _ = load_balanced(C, ds, seed, lowmem=lowmem)
            st = C.run_static_eval(X_tr, y_tr, X_val, y_val, X_te, y_te, NC, ds)
            mm, md, wt, wd, thr = st["m_main"], st["m_div"], st["wt"], st["wd"], st["thr"]
            for method in methods:
                if method == "Static XGBoost":
                    f1s = [st["f1m"]] * C.N_BATCHES; fprs = [st["fpr"]] * C.N_BATCHES
                elif method == "Fixed Replay (ER)":
                    r = run_er(C, mm, md, wt, wd, thr, X_tr, y_tr, X_stream, y_stream, X_te, y_te, NC, seed)
                    f1s = [x["f1_macro"] for x in r]; fprs = [x["fpr_macro"] for x in r]
                elif method in ("PSI-only", "CADENCE"):
                    r = run_cl_tuned(C, mm, md, wt, wd, thr, X_tr, y_tr, X_val, y_val, X_stream, y_stream, X_te, y_te, NC, seed, use_context=(method == "CADENCE"), guarded=(not args.no_guard))
                    f1s = [x["f1_macro"] for x in r]; fprs = [x["fpr_macro"] for x in r]
                elif method in ("EWC", "LwF"):
                    r = run_ewc_lwf(C, method, X_tr, y_tr, X_val, y_val, X_stream, y_stream, X_te, y_te, NC, seed)
                    f1s = [x["f1_macro"] for x in r]; fprs = [x["fpr_macro"] for x in r]
                elif method == "DER":
                    r = run_der(C, X_tr, y_tr, X_val, y_val, X_stream, y_stream, X_te, y_te, NC, seed)
                    f1s = [x["f1_macro"] for x in r]; fprs = [x["fpr_macro"] for x in r]
                else:
                    continue
                m = C.cl_metrics(f1s, fprs)
                rows.append(dict(dataset=ds, method=method, seed=seed, CL_F1=m["CL_F1"],
                                 Forgetting=m["Forgetting"], DriftRes=m["DriftRes"],
                                 Stability=m["Stability"], CL_FPR=m["CL_FPR"]))
                pd.DataFrame(rows).to_csv(args.out, index=False)
                print("    %-18s CL-F1=%.4f  Forget=%.4f  FPR=%.4f" % (method, m["CL_F1"], m["Forgetting"], m["CL_FPR"]), flush=True)
            print("  (%.0fs)" % (time.time() - t0), flush=True); gc.collect()

    df = pd.DataFrame(rows); df.to_csv(args.out, index=False)
    print("\n[saved] %s\n\n=========  MEAN +/- STD  =========" % args.out)
    g = df.groupby(["dataset", "method"]).agg(F1=("CL_F1", "mean"), SD=("CL_F1", "std"),
                                              FG=("Forgetting", "mean"), FPR=("CL_FPR", "mean")).reset_index()
    for ds in df.dataset.unique():
        print("\n  %s" % ds)
        sub = g[g.dataset == ds].sort_values("F1", ascending=False)
        for _, r in sub.iterrows():
            sd = 0.0 if pd.isna(r.SD) else r.SD
            best = "  <== best" if r.method == sub.iloc[0].method else ""
            print("    %-18s CL-F1 %.4f +/- %.4f  Forget %.4f  FPR %.4f%s" % (r.method, r.F1, sd, r.FG, r.FPR, best))


if __name__ == "__main__":
    main()
