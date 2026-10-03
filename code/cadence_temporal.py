#!/usr/bin/env python3
# CADENCE TEMPORAL-DRIFT BENCHMARK (REAL DATA ORDER, NO SHUFFLE)
# Streams follow the natural chronological/phase order of each dataset (genuine
# concept drift). Prequential (test-then-train) evaluation: every batch is scored
# BEFORE the model adapts on it, so a frozen model degrades as drift accrues while
# an adaptive model recovers. Methods: Static, PSI-only, CADENCE (+ ablation).
#   python cadence_temporal.py --data-root ./data --datasets TON_IoT CICIDS2017 Edge_IIoTset UNSW_NB15 IoT23 MQTTset
from __future__ import annotations
import argparse, importlib.util, os, sys, gc, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

def load_core(p):
    s=importlib.util.spec_from_file_location("cadence_core",os.path.abspath(p)); m=importlib.util.module_from_spec(s)
    sys.modules["cadence_core"]=m; s.loader.exec_module(m); return m
def load_runner(p):
    s=importlib.util.spec_from_file_location("cadence_runner",os.path.abspath(p)); m=importlib.util.module_from_spec(s)
    sys.modules["cadence_runner"]=m; s.loader.exec_module(m); return m

TIME_COL_CANDIDATES=["Timestamp","timestamp"," Timestamp","Stime","stime","ts","Ltime","ltime","time","Time","frame.time","date","Date"]

def detect_time_col(df, time_col=None):
    """Return the name of a usable timestamp column, or None."""
    cands=[time_col] if time_col else TIME_COL_CANDIDATES
    for c in cands:
        if c is not None and c in df.columns:
            return c
    return None

def parse_time(series):
    """Numeric epoch stays numeric; strings are parsed day-first (CIC-IDS2017 style)."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series,errors="coerce")
    t=pd.to_datetime(series.astype(str).str.strip(),errors="coerce",dayfirst=True)
    return t.astype("int64",errors="ignore") if t.notna().any() else pd.to_numeric(series,errors="coerce")

def load_temporal(C, R, ds, seed, init_frac=0.45, order="file", time_col=None):
    """Order-preserving balanced sample -> init (earliest) + ordered stream.
    order='file' : rows in released-file order (default; the v3/v4 streams)
    order='time' : rows sorted by a timestamp column (auto-detected or --time-col);
                   raises if no usable column exists so the caller can report it."""
    cfg=C.DS_CONFIGS[ds]; path=R._csv_path(C,cfg); lab=cfg["label_col"]; budget=cfg["n_samples"]
    df=pd.read_csv(path,low_memory=False).reset_index(drop=True)
    if order=="time":
        tc=detect_time_col(df,time_col)
        if tc is None:
            raise ValueError("%s: no timestamp column found in %s (columns tried: %s)" % (ds, os.path.basename(path), ", ".join(TIME_COL_CANDIDATES)))
        tv=parse_time(df[tc]); nbad=int(pd.isna(tv).sum())
        df["__t"]=tv; df=df.sort_values("__t",kind="stable",na_position="last").drop(columns="__t").reset_index(drop=True)
        print("  [order=time] %s sorted by '%s' (%d rows, %d unparseable placed last)" % (ds, tc, len(df), nbad), flush=True)
    df["__ord"]=np.arange(len(df))
    cnt={str(k):int(v) for k,v in df[lab].astype(str).value_counts().items()}
    take=R._waterfill(cnt,budget)
    parts=[g.sample(n=min(take.get(str(c),2),len(g)),random_state=seed) for c,g in df.groupby(df[lab].astype(str),sort=False)]
    s=pd.concat(parts).sort_values("__ord").reset_index(drop=True); s=s.drop(columns="__ord"); del df,parts; gc.collect()
    le=C.LabelEncoder(); y=le.fit_transform(s[lab].astype(str))
    drop=[c for c in cfg["drop_cols"]+[lab] if c in s.columns]; s=s.drop(columns=drop)
    for col in s.columns:
        if s[col].dtype==object: s[col]=C.LabelEncoder().fit_transform(s[col].astype(str).fillna("NA"))
    for col in s.columns: s[col]=pd.to_numeric(s[col],errors="coerce").fillna(0)
    X=np.nan_to_num(s.values.astype(np.float32)); NC=int(len(le.classes_)); del s; gc.collect()
    n=len(X); i1=int(n*init_frac); X_init,y_init=X[:i1],y[:i1]; X_str,y_str=X[i1:],y[i1:]
    nzv=X_init.var(0)>1e-6; X_init,X_str=X_init[:,nzv],X_str[:,nzv]
    sc=C.RobustScaler(quantile_range=(5,95)).fit(X_init)
    X_init=np.nan_to_num(sc.transform(X_init).astype(np.float32)); X_str=np.nan_to_num(sc.transform(X_str).astype(np.float32))
    from sklearn.model_selection import train_test_split
    try: Xtr,Xv,ytr,yv=train_test_split(X_init,y_init,test_size=0.15,random_state=seed,stratify=y_init)
    except Exception: Xtr,Xv,ytr,yv=train_test_split(X_init,y_init,test_size=0.15,random_state=seed)
    return Xtr,Xv,ytr,yv,X_str,y_str,NC

def prequential(C, ds, seed, Xtr,Xv,ytr,yv,X_str,y_str,NC, mode, T=8, return_stages=False):
    """CADENCE = guarded EAGER continual adaptation. The model adapts on every
    drifting stage and the validation guard decides what to keep; the Context
    engine modulates HOW each update is formed (minority-oversampling boost and
    threshold calibration). PSI is an optional compute-efficiency layer that may
    skip stages it deems stable.
    Modes:
        static         : no adaptation (frozen reference)
        full_cadence   : guarded eager adaptation + context-modulated retraining
        wo_context     : guarded eager adaptation, fixed retrain params (no context)
        wo_guard       : eager adaptation + context, no validation guard
        psi_efficiency : PSI-gated adaptation + context + guard (low-compute variant)
    Legacy aliases: wo_psi->full_cadence, cadence->psi_efficiency,
                    psi->wo_context(PSI-gated), cadence_noguard->PSI-gated,no-guard.
    """
    import xgboost as xgb
    from sklearn.model_selection import train_test_split as _tts
    from sklearn.metrics import f1_score
    st=C.run_static_eval(Xtr,ytr,Xv,yv,Xv,yv,NC,ds)
    mm,md,wt,wd,thr=st["m_main"],st["m_div"],st["wt"],st["wd"],st["thr"]
    EAGER={"wo_psi","full_cadence","wo_context","wo_guard"}
    use_context  = mode in ("cadence","wo_psi","full_cadence","wo_guard","psi_efficiency")
    guarded      = mode not in ("cadence_noguard","wo_guard")
    always_adapt = mode in EAGER
    def train_b(Xb,yb,rounds,mb=2.5,tm=1.0):
        # context modulates the minority-oversampling ratio and threshold scale
        ratio=float(np.clip(0.55*(mb/2.5),0.35,0.85))
        Xa,ya=C.smote_safe(Xb,yb,NC,ratio=ratio,k=3,severe=False,seed=seed); bw=C.eff_num_weights(ya)
        m1=C._train_xgb(Xa,ya,Xv,yv,NC,bw,"main",rounds); p=m1.predict(xgb.DMatrix(Xa)).reshape(-1,NC).astype(np.float32)
        m2=C._train_xgb(Xa,ya,Xv,yv,NC,C.focal_proxy_weights(ya,p,gamma=1.5,base_w=bw),"div",rounds)
        w1,w2=C._val_weights(m1,m2,Xv,yv,NC)
        th=C.calibrate_thresholds(C._ensemble_probs(m1,m2,Xv,NC,w1,w2),yv,NC,30)
        th=np.clip(th*tm,1e-3,0.99).astype(np.float32)
        return m1,m2,w1,w2,th
    drift=C.AdaptiveDriftDetector(); ctx=C.ContextStateEngine(Xtr,ytr,NC) if use_context else None; pe=C.ContextGatedPolicy() if use_context else None
    batches=np.array_split(np.arange(len(y_str)),T); f1s=[]; last=-9
    for bi,idx in enumerate(batches):
        Xb,yb=X_str[idx],y_str[idx]
        yp=C.apply_thresholds(C._ensemble_probs(mm,md,Xb,NC,wt,wd),thr); f1s.append(f1_score(yb,yp,average="macro",zero_division=0))
        if mode=="static": continue
        di=drift.update(Xb,mean_conf=float(C._ensemble_probs(mm,md,Xb,NC,wt,wd).max(1).mean())); pint=drift.retrain_intensity()
        mb,tm=2.5,1.0
        if use_context:
            cs,csc,cst=ctx.analyse(Xb,yb,di["severity"])
            pol=pe.get(cs,di["state"],pint,csc,cst["ctx_score_trend"],cst["n_classes_seen"],cst["class_entropy"],cst["imbalance"])
            inten=pol.intensity; mb,tm=pol.minority_boost,pol.thr_mult
        else: inten=pint
        if always_adapt: inten="full"       # eager: adapt every stage, guard decides
        # cooldown: step a recent "full" down to "partial" but never to "skip"
        if (bi-last)<1 and inten=="full": inten="partial"
        if inten!="skip":
            rounds=130 if inten=="full" else 90
            mn=np.bincount(yb,minlength=NC).min()
            try: Xrt,Xrv,yrt,yrv=_tts(Xb,yb,test_size=0.25,random_state=seed,stratify=yb if mn>=2 else None)
            except Exception: Xrt,Xrv,yrt,yrv=_tts(Xb,yb,test_size=0.25,random_state=seed)
            cm,cd,cw,cwd2,cth=train_b(Xrt,yrt,rounds,mb,tm)
            def rvf(a,b,c,d,e):
                from sklearn.metrics import f1_score as _f
                return _f(yrv,C.apply_thresholds(C._ensemble_probs(a,b,Xrv,NC,c,d),e),average="macro",zero_division=0)
            if (not guarded) or (rvf(cm,cd,cw,cwd2,cth)>=rvf(mm,md,wt,wd,thr)):
                mm,md,wt,wd,thr=cm,cd,cw,cwd2,cth
            last=bi
    return (list(map(float,f1s)) if return_stages else float(np.mean(f1s)))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--core",default="cadence_core.py"); ap.add_argument("--runner",default="cadence_runner.py")
    ap.add_argument("--data-root",default="./data")
    ap.add_argument("--datasets",nargs="+",default=["TON_IoT","CICIDS2017","Edge_IIoTset","UNSW_NB15","IoT23","MQTTset"])
    ap.add_argument("--seeds",nargs="+",type=int,default=[42,123,456])
    ap.add_argument("--out",default="temporal_results.csv")
    a=ap.parse_args()
    C=load_core(a.core); R=load_runner(a.runner); R.remap_paths(C,a.data_root)
    modes=[("Static","static"),
           ("PSI-efficiency","psi_efficiency"),
           ("w/o Guard","wo_guard"),
           ("Full CADENCE","full_cadence")]
    rows=[]
    for ds in a.datasets:
        if ds not in C.DS_CONFIGS:
            print("skip",ds); continue
        for seed in a.seeds:
            print("\n==== %s seed=%d (temporal/prequential) ====" % (ds,seed),flush=True)
            Xtr,Xv,ytr,yv,X_str,y_str,NC=load_temporal(C,R,ds,seed)
            for name,mode in modes:
                v=prequential(C,ds,seed,Xtr,Xv,ytr,yv,X_str,y_str,NC,mode)
                rows.append(dict(dataset=ds,seed=seed,method=name,preq_CL_F1=round(v,4)))
                pd.DataFrame(rows).to_csv(a.out,index=False)
                print("    %-16s preq CL-F1=%.4f"%(name,v),flush=True)
            del Xtr,Xv,X_str; gc.collect()
    df=pd.DataFrame(rows); df.to_csv(a.out,index=False)
    print("\n[saved] %s\n\n===== TEMPORAL (mean prequential CL-F1) ====="%a.out)
    g=pd.DataFrame(rows).groupby(["dataset","method"])["preq_CL_F1"].mean().reset_index()
    print(g.to_string(index=False))

if __name__=="__main__":
    main()
