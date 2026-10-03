#!/usr/bin/env python3
# =============================================================================
# CADENCE-H : VALIDATION-GATED MIXTURE OF BATCH AND ONLINE EXPERTS
#
# The guard set G_t is no longer a binary accept/reject switch only.  It also
# sets a CONTINUOUS trust weight between two experts with complementary
# failure modes:
#
#     F_B  batch expert   : XGBoost dual ensemble, retrained per (sub-)stage on
#                           U_t (+ replay), admitted only if it does not lower
#                           macro-F1 on G_t           -> strong on gradual drift
#     F_O  online expert  : Adaptive Random Forest, learns per sample after the
#                           chunk has been scored     -> strong on class emergence
#
#     p_t(y|x) = w_t * p_B(y|x) + (1 - w_t) * p_O(y|x)
#
#     w_t = EMA( softmax( beta * [ phi_B(t), phi_O(t) ] )[0] )
#
# where phi_B(t), phi_O(t) are each expert's PREQUENTIAL macro-F1 on the whole
# stage t, computed from predictions SAVED BEFORE the stage's labels were
# released (default --gate prequential).  G_t is used ONLY for binary admission
# of the batch candidate.  (--gate guard scores both experts on G_t after
# learning U_t instead; kept as an ablation.)  w_t is applied to stage t+1.
#
# EVENT ORDER PER STAGE (identical for every method):
#   1. predict the whole stage with frozen models; SAVE predictions/scores
#   2. labels released
#   3. metrics (CL-F1, phi_B, phi_O, control mixtures) from the SAVED predictions
#   4. partition stage -> U_t (75%) | G_t (25%)
#   5. batch candidate on U_t (+replay); admission on G_t; online expert learns U_t
#   6. next-stage weight from step-3 scores; online expert learns G_t; buffer
#   7. retention on X_val and X_hist
#
# LABEL TIMING (identical for every method, including the online experts):
#   each chunk is scored in full by the current predictor, THEN labels are
#   released and every method updates.  No method learns from a sample before
#   predicting it.  (The paper's earlier PWPAE baseline interleaved
#   predict/learn per sample; PWPAE-fair below corrects that.)
#
# LEAKAGE-FREE GUARD: each chunk is partitioned once into U_t (75%, training)
#   and G_t (25%, guard).  Batch candidates never see G_t.
#
# INDEPENDENT HISTORICAL TEST: X_hist is carved out of the initialization
#   training portion and is never used for training, calibration or admission.
#
# Methods
#   Static-XGB   no update
#   CADENCE-B    U_t only,          guarded          (batch-only guard arm)
#   ER-XGB       U_t + buffer,      always accept
#   CADENCE-R    U_t + buffer,      guarded
#   ARF          online only, fair timing
#   PWPAE-fair   PWPAE ensemble,    fair timing
#   CADENCE-H    mixture of CADENCE-R (F_B) and ARF (F_O), prequentially weighted
#   Naive-XGB    U_t only,          always accept    (isolates admission vs CADENCE-B)
#   CADENCE-H-ng CADENCE-H with F_B always accepted   (isolates admission inside H)
#
# v4 arms (protocol A/Bs and the combination-rule benchmark):
#   PWPAE-interleaved  same PWPAE members, but predict-then-learn PER SAMPLE inside
#                      each stage (the earlier paper's schedule). Matched against
#                      PWPAE-fair: the ONLY variable is the label-release schedule.
#   CADENCE-R-leaky    identical candidate to CADENCE-R (fit on U_t + replay), but the
#                      admission slice is drawn FROM U_t, i.e. from records the
#                      candidate trained on. Matched against CADENCE-R: the ONLY
#                      variable is whether the guard overlaps the training data.
#   DWM                Kolter-Maloof Dynamic Weighted Majority over the SAME two
#                      experts as CADENCE-H: per-record correctness on the saved
#                      predictions multiplies an expert's weight by beta_dwm (0.5)
#                      once per stage after labels; weighted label vote. The parent
#                      rule of the prequential gate, run as a matched arm.
#
# v4 columns on every mixture row (CADENCE-H, CADENCE-H-ng, DWM), from saved scores:
#   f1_fix00 .. f1_fix10 : CL-F1 of a fixed weight w in {0.0,0.1,...,1.0}
#                          (fix00 = online alone, fix10 = batch alone)
#   disagree             : fraction of records where the two experts' argmax differ
#   oracle_union         : fraction of records at least one expert classifies correctly
# v4 columns on every row: n_guard, guard_min_class, sec_score, sec_fit
#
# CONTROLS logged on every CADENCE-H row from the SAME saved expert scores:
#   f1_fix50   : fixed w=0.5 mixture
#   f1_fixcal  : fixed w = w_0 (calibration-selected, never updated)
#   f1_hard    : hard selection of the expert that won the previous stage
#   w_raw      : softmax weight before smoothing; w = smoothed weight USED here
#
# Usage (server, Linux):
#   python3 cadence_hybrid.py --data-root ./data --seeds 42 123 456 --out hybrid_results.csv
# Smoke test:
#   python3 cadence_hybrid.py --data-root ./data --datasets IoT23 --seeds 42 --budget 8000 \
#       --methods Static-XGB CADENCE-R ARF CADENCE-H --out hybrid_smoke.csv
# Finer granularity (k sub-stages per stage; full retrain per sub-stage):
#   add  --substages 2
# =============================================================================
from __future__ import annotations
import argparse, importlib.util, os, sys, gc, warnings, time
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

ALL_METHODS = ["Static-XGB", "Naive-XGB", "CADENCE-B", "ER-XGB", "CADENCE-R",
               "ARF", "PWPAE-fair", "CADENCE-H", "CADENCE-H-ng",
               "PWPAE-interleaved", "CADENCE-R-leaky", "DWM"]
# rerun-2 arms: the same three online-expert arms with Streaming Random Patches as F_O
SRP_METHODS = ["SRP", "CADENCE-H-srp", "DWM-srp"]
# rerun-2 dataset: raw CIC-IDS2017 flows merged in capture order (see prepare_cicids_raw.py)
CHRONO_CONFIG = dict(display="CICIDS2017 (chronological)",
                     path="cadence/datasets/CICIDS2017_chrono/cicids2017_chrono.csv",
                     label_col="Attack Type", n_samples=22_000,
                     drop_cols=["Flow ID", "Source IP", "Destination IP", "Source Port", "Timestamp"])
FIXW = [round(0.1 * i, 1) for i in range(11)]          # fixed-weight sweep


def load(p, name):
    s = importlib.util.spec_from_file_location(name, os.path.abspath(p))
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


# ---------------- metrics ---------------------------------------------------
def macro_f1(y, yp):
    from sklearn.metrics import f1_score
    return float(f1_score(y, yp, average="macro", zero_division=0))


def macro_fpr(y, yp, nc):
    y = np.asarray(y); yp = np.asarray(yp); out = []
    for c in range(nc):
        neg = (y != c)
        if neg.sum() == 0:
            continue
        fp = ((yp == c) & neg).sum(); tn = (neg & (yp != c)).sum()
        out.append(fp / max(1, fp + tn))
    return float(np.mean(out)) if out else 0.0


def split_chunk(Xb, yb, NC, seed, guard_frac=0.25):
    """Partition ONCE into U (train) and G (guard). Candidates never see G."""
    from sklearn.model_selection import train_test_split as tts
    mn = np.bincount(yb, minlength=NC).min()
    try:
        return tts(Xb, yb, test_size=guard_frac, random_state=seed,
                   stratify=yb if mn >= 2 else None)
    except Exception:
        return tts(Xb, yb, test_size=guard_frac, random_state=seed)


# ---------------- batch expert (identical CADENCE XGBoost backbone) ---------
class BatchExpert:
    """XGBoost dual ensemble. Calibration on X_val only. Returns normalized,
    threshold-adjusted class scores so it can be mixed with the online expert."""
    def __init__(self, C, Xv, yv, NC, seed, rounds):
        self.C, self.Xv, self.yv, self.NC, self.seed, self.rounds = C, Xv, yv, NC, seed, rounds
        self.M = None

    def train(self, X, y):
        import xgboost as xgb
        C, NC = self.C, self.NC
        Xa, ya = C.smote_safe(X, y, NC, ratio=0.55, k=3, severe=False, seed=self.seed)
        bw = C.eff_num_weights(ya)
        m1 = C._train_xgb(Xa, ya, self.Xv, self.yv, NC, bw, "main", self.rounds)
        p = m1.predict(xgb.DMatrix(Xa)).reshape(-1, NC).astype(np.float32)
        m2 = C._train_xgb(Xa, ya, self.Xv, self.yv, NC,
                          C.focal_proxy_weights(ya, p, gamma=1.5, base_w=bw),
                          "div", self.rounds)
        w1, w2 = C._val_weights(m1, m2, self.Xv, self.yv, NC)
        th = C.calibrate_thresholds(C._ensemble_probs(m1, m2, self.Xv, NC, w1, w2),
                                    self.yv, NC, 30)
        return (m1, m2, w1, w2, th)

    def scores(self, M, X):
        """Threshold-adjusted, row-normalized scores in [0,1]^NC."""
        m1, m2, w1, w2, th = M
        p = self.C._ensemble_probs(m1, m2, X, self.NC, w1, w2)
        s = p / (np.asarray(th, dtype=np.float32) + 1e-9)
        s = s / np.maximum(s.sum(1, keepdims=True), 1e-12)
        return s.astype(np.float32)

    def predict(self, M, X):
        return self.scores(M, X).argmax(1)

    def f1(self, M, X, y):
        return macro_f1(y, self.predict(M, X))


# ---------------- online expert (river ARF, fair timing) --------------------
class OnlineExpert:
    def __init__(self, NC, seed, n_models=10, kind="arf"):
        from river import forest
        self.NC = NC
        if kind == "srp":
            from river import ensemble, tree
            self.m = ensemble.SRPClassifier(model=tree.HoeffdingTreeClassifier(grace_period=50),
                                            n_models=n_models, seed=seed)
        else:
            self.m = forest.ARFClassifier(n_models=n_models, seed=seed)

    @staticmethod
    def _d(x):
        return {i: float(v) for i, v in enumerate(x)}

    def learn(self, X, y):
        for x, yy in zip(X, y):
            self.m.learn_one(self._d(x), int(yy))

    def scores(self, X):
        out = np.zeros((len(X), self.NC), dtype=np.float32)
        for i, x in enumerate(X):
            try:
                pr = self.m.predict_proba_one(self._d(x)) or {}
            except Exception:
                pr = {}
            if pr:
                for k, v in pr.items():
                    k = int(k)
                    if 0 <= k < self.NC:
                        out[i, k] = v
            else:
                out[i, :] = 1.0 / self.NC
        s = out.sum(1, keepdims=True)
        return out / np.maximum(s, 1e-12)

    def predict(self, X):
        return self.scores(X).argmax(1)

    def f1(self, X, y):
        return macro_f1(y, self.predict(X))


# ---------------- PWPAE with FAIR (batch-delayed) label timing --------------
class PWPAEFair:
    def __init__(self, NC, seed):
        from river import forest, ensemble, tree, drift
        self.NC = NC
        self.members = [
            ("ARF-ADWIN", forest.ARFClassifier(n_models=3, seed=seed, drift_detector=drift.ADWIN())),
            ("ARF-PH",    forest.ARFClassifier(n_models=3, seed=seed, drift_detector=drift.PageHinkley())),
            ("SRP-ADWIN", ensemble.SRPClassifier(model=tree.HoeffdingTreeClassifier(grace_period=50),
                                                  n_models=3, seed=seed, drift_detector=drift.ADWIN())),
            ("SRP-PH",    ensemble.SRPClassifier(model=tree.HoeffdingTreeClassifier(grace_period=50),
                                                  n_models=3, seed=seed, drift_detector=drift.PageHinkley())),
        ]
        self.acc = {n: [1e-6, 1e-6] for n, _ in self.members}

    @staticmethod
    def _d(x):
        return {i: float(v) for i, v in enumerate(x)}

    def learn(self, X, y):
        for x, yy in zip(X, y):
            xi = self._d(x)
            for n, mdl in self.members:
                mdl.learn_one(xi, int(yy))

    def predict(self, X, y_for_acc=None):
        """Score the WHOLE chunk with the current members, then (optionally)
        update running accuracies.  No member learns inside this call."""
        preds = np.zeros(len(X), dtype=int)
        member_preds = {n: [] for n, _ in self.members}
        for i, x in enumerate(X):
            xi = self._d(x); agg = {}
            for n, mdl in self.members:
                w = self.acc[n][0] / self.acc[n][1]
                try:
                    pr = mdl.predict_proba_one(xi) or {}
                except Exception:
                    pr = {}
                for k, v in pr.items():
                    agg[k] = agg.get(k, 0.0) + w * v
                member_preds[n].append(int(max(pr, key=pr.get)) if pr else -1)
            preds[i] = int(max(agg, key=agg.get)) if agg else 0
        if y_for_acc is not None:                     # performance weights,
            for n, _ in self.members:                 # updated AFTER scoring
                mp = np.asarray(member_preds[n])
                self.acc[n][1] += len(mp)
                self.acc[n][0] += int((mp == np.asarray(y_for_acc)).sum())
        return preds

    def f1(self, X, y):
        return macro_f1(y, self.predict(X))

    def predict_learn_interleaved(self, X, y):
        """Per-sample test-then-train INSIDE the stage: predict x_i, then every
        member learns (x_i, y_i) and its running accuracy is updated, before
        x_{i+1} is predicted.  This is the schedule the earlier paper used for
        PWPAE and is the A/B partner of predict()."""
        preds = np.zeros(len(X), dtype=int)
        for i, (x, yy) in enumerate(zip(X, y)):
            xi = self._d(x); agg = {}
            for n, mdl in self.members:
                w = self.acc[n][0] / self.acc[n][1]
                try:
                    pr = mdl.predict_proba_one(xi) or {}
                except Exception:
                    pr = {}
                for k, v in pr.items():
                    agg[k] = agg.get(k, 0.0) + w * v
                mp = int(max(pr, key=pr.get)) if pr else -1
                self.acc[n][1] += 1; self.acc[n][0] += int(mp == int(yy))
            preds[i] = int(max(agg, key=agg.get)) if agg else 0
            for n, mdl in self.members:
                mdl.learn_one(xi, int(yy))
        return preds


# ---------------- exemplar buffer -------------------------------------------
class Buffer:
    def __init__(self, cap_per_class, seed):
        self.cap = cap_per_class; self.rng = np.random.default_rng(seed)
        self.X = None; self.y = None

    def add(self, X, y):
        self.X = X.copy() if self.X is None else np.vstack([self.X, X])
        self.y = y.copy() if self.y is None else np.concatenate([self.y, y])
        keep = []
        for c in np.unique(self.y):
            ci = np.where(self.y == c)[0]
            if len(ci) > self.cap:
                ci = self.rng.choice(ci, self.cap, replace=False)
            keep.append(ci)
        k = np.concatenate(keep); self.X = self.X[k]; self.y = self.y[k]

    def get(self):
        return (self.X, self.y) if self.X is not None else (None, None)


# ---------------- the gate --------------------------------------------------
def gate_weight(f1_B, f1_O, beta):
    z = np.array([f1_B, f1_O], dtype=np.float64) * beta
    z -= z.max()
    e = np.exp(z)
    return float(e[0] / e.sum())


def dwm_vote(sB, sO, lwB, lwO):
    """Weighted label vote of two experts (DWM). Ties go to the batch expert."""
    yB, yO = sB.argmax(1), sO.argmax(1)
    wB, wO = np.exp(lwB), np.exp(lwO)
    return np.where(wB >= wO, yB, yO)


# ---------------- one run of one method --------------------------------------
def run_method(C, method, seed, a, Xtr, ytr, Xv, yv, Xhist, yhist, X_str, y_str, NC):
    T, k = 8, a.substages
    rows = []
    BE = BatchExpert(C, Xv, yv, NC, seed, a.rounds)

    # ---- initialise experts ------------------------------------------------
    is_H = method in ("CADENCE-H", "CADENCE-H-ng", "CADENCE-H-srp")
    is_DWM = method in ("DWM", "DWM-srp")
    is_mix = is_H or is_DWM
    is_PW = method in ("PWPAE-fair", "PWPAE-interleaved")
    is_online_only = method in ("ARF", "SRP")
    online_kind = "srp" if (method == "SRP" or method.endswith("-srp")) else "arf"
    MB = BE.train(Xtr, ytr) if method not in ("ARF", "SRP", "PWPAE-fair", "PWPAE-interleaved") else None
    OE = PE = None
    if is_online_only or is_mix:
        OE = OnlineExpert(NC, seed, n_models=a.arf_models, kind=online_kind)
        idx = np.arange(len(ytr))
        if len(idx) > a.warm:
            idx = np.random.default_rng(seed).choice(idx, a.warm, replace=False)
        OE.learn(Xtr[idx], ytr[idx])
    if is_PW:
        PE = PWPAEFair(NC, seed)
        idx = np.arange(len(ytr))
        if len(idx) > a.warm:
            idx = np.random.default_rng(seed).choice(idx, a.warm, replace=False)
        PE.learn(Xtr[idx], ytr[idx])

    buf = Buffer(a.buffer_per_class, seed); buf.add(Xtr, ytr)
    guarded = method in ("CADENCE-B", "CADENCE-R", "CADENCE-H", "CADENCE-R-leaky", "DWM", "CADENCE-H-srp", "DWM-srp")
    leaky = method == "CADENCE-R-leaky"
    use_buf = method in ("ER-XGB", "CADENCE-R", "CADENCE-H", "CADENCE-H-ng", "CADENCE-R-leaky", "DWM", "CADENCE-H-srp", "DWM-srp")
    lwB = lwO = 0.0                        # DWM log-weights (start equal)

    # initial gate weight from the calibration split (not test data)
    w = 1.0; w_cal = None; prev_winner = None
    if is_mix:
        w = gate_weight(BE.f1(MB, Xv, yv), OE.f1(Xv, yv), a.beta)
        w_cal = w                          # fixed calibration-selected weight (control)
        prev_winner = "B" if w >= 0.5 else "O"

    stages = np.array_split(np.arange(len(y_str)), T)
    for bi, sidx in enumerate(stages):
        chunks = np.array_split(sidx, k)
        for ci, idx in enumerate(chunks):
            Xb, yb = X_str[idx], y_str[idx]
            rec = dict(stage=bi, sub=ci, n=len(yb))

            # ---- 1. SCORE the chunk with the current predictor (no learning) ----
            f1B = f1O = np.nan; ctrl = {}
            t_score = time.time()
            if method in ("Static-XGB", "Naive-XGB", "CADENCE-B", "ER-XGB", "CADENCE-R", "CADENCE-R-leaky"):
                yp = BE.predict(MB, Xb)
            elif is_online_only:
                yp = OE.predict(Xb)
            elif method == "PWPAE-fair":
                yp = PE.predict(Xb, y_for_acc=yb)
            elif method == "PWPAE-interleaved":
                yp = PE.predict_learn_interleaved(Xb, yb)        # learns INSIDE scoring
            elif is_mix:
                sB = BE.scores(MB, Xb); sO = OE.scores(Xb)          # SAVED scores
                if is_DWM:
                    yp = dwm_vote(sB, sO, lwB, lwO)
                else:
                    yp = (w * sB + (1.0 - w) * sO).argmax(1)
                # ---- labels released below: metrics from the SAVED scores ----
                yB_, yO_ = sB.argmax(1), sO.argmax(1)
                f1B = macro_f1(yb, yB_); f1O = macro_f1(yb, yO_)
                ctrl = dict(
                    f1_fix50=macro_f1(yb, (0.5 * sB + 0.5 * sO).argmax(1)),
                    f1_fixcal=macro_f1(yb, (w_cal * sB + (1 - w_cal) * sO).argmax(1)) if w_cal is not None else np.nan,
                    f1_hard=(f1B if prev_winner == "B" else f1O),
                    changed_vs_best=float(np.mean(yp != (sB if f1B >= f1O else sO).argmax(1))),
                    disagree=float(np.mean(yB_ != yO_)),
                    oracle_union=float(np.mean((yB_ == yb) | (yO_ == yb))),
                )
                for fw in FIXW:
                    ctrl["f1_fix%02d" % int(round(fw * 10))] = macro_f1(yb, (fw * sB + (1 - fw) * sO).argmax(1))
            rec["sec_score"] = time.time() - t_score
            rec.update(f1=macro_f1(yb, yp), fpr=macro_fpr(yb, yp, NC),
                       f1_B=f1B, f1_O=f1O,
                       w=(w if is_H else (float(np.exp(lwB) / (np.exp(lwB) + np.exp(lwO))) if is_DWM else np.nan)),
                       **ctrl)

            # ---- 2. labels released: partition chunk, build candidate -----------
            Xu, Xg, yu, yg = split_chunk(Xb, yb, NC, seed)
            rec["n_guard"] = int(len(yg)); rec["guard_min_class"] = int(np.bincount(yg, minlength=NC)[np.bincount(yg, minlength=NC) > 0].min())
            accepted = None; attempted = False
            t_fit = time.time()
            if method in ("Naive-XGB", "CADENCE-B", "ER-XGB", "CADENCE-R", "CADENCE-H", "CADENCE-H-ng",
                          "CADENCE-R-leaky", "DWM", "CADENCE-H-srp", "DWM-srp"):
                attempted = True
                if use_buf:
                    bX, bY = buf.get()
                    cand = BE.train(np.vstack([Xu, bX]), np.concatenate([yu, bY]))
                else:
                    cand = BE.train(Xu, yu)
                if guarded:
                    if leaky:
                        # admission slice drawn FROM the candidate's own training
                        # records (same size as G_t): the earlier paper's overlap.
                        _, Xg_l, _, yg_l = split_chunk(Xu, yu, NC, seed, guard_frac=min(0.99, len(yg) / max(1, len(yu))))
                        accepted = BE.f1(cand, Xg_l, yg_l) >= BE.f1(MB, Xg_l, yg_l)
                    else:
                        accepted = BE.f1(cand, Xg, yg) >= BE.f1(MB, Xg, yg)
                    if accepted:
                        MB = cand
                else:
                    MB = cand; accepted = True

            # ---- 3. online expert learns U_t (G_t still unseen by everyone) ----
            if OE is not None:
                OE.learn(Xu, yu)
            if PE is not None and method == "PWPAE-fair":
                PE.learn(Xu, yu)

            # ---- 4. gate ------------------------------------------------------
            #   --gate prequential : each expert's F1 on this chunk BEFORE it learned
            #                        it (step 1).  Measures adaptation to novelty
            #                        directly; leakage-free by construction.
            #   --gate guard       : both experts scored on the unseen G_t after
            #                        learning U_t.  Measures within-stage fit and
            #                        lags one stage under class emergence.
            if is_H:
                if a.gate == "prequential":
                    fB, fO = f1B, f1O
                else:
                    fB, fO = BE.f1(MB, Xg, yg), OE.f1(Xg, yg)
                w_new = gate_weight(fB, fO, a.beta)
                w = a.w_ema * w_new + (1.0 - a.w_ema) * w
                prev_winner = "B" if fB >= fO else "O"
                rec.update(gate_f1_B=fB, gate_f1_O=fO, w_raw=w_new, w_next=w)
            if is_DWM:
                # Kolter-Maloof update from the SAVED predictions, once per stage:
                # each wrong record multiplies the expert's weight by beta_dwm.
                nB = int((sB.argmax(1) != yb).sum()); nO = int((sO.argmax(1) != yb).sum())
                lwB += nB * np.log(a.beta_dwm); lwO += nO * np.log(a.beta_dwm)
                m_ = max(lwB, lwO); lwB -= m_; lwO -= m_          # normalise (log-space)
                prev_winner = "B" if f1B >= f1O else "O"       # for the f1_hard control
                rec.update(dwm_wrong_B=nB, dwm_wrong_O=nO, w_next=float(np.exp(lwB) / (np.exp(lwB) + np.exp(lwO))))

            # ---- 5. now the online experts may learn G_t as well -------------
            if OE is not None:
                OE.learn(Xg, yg)
            if PE is not None and method == "PWPAE-fair":
                PE.learn(Xg, yg)
            buf.add(Xb, yb)
            rec["sec_fit"] = time.time() - t_fit

            # ---- 6. retention: protected set vs INDEPENDENT historical set ----
            if is_online_only:
                rv, rh = OE.f1(Xv, yv), OE.f1(Xhist, yhist)
            elif is_PW:
                rv, rh = PE.f1(Xv, yv), PE.f1(Xhist, yhist)
            elif is_DWM:
                def mixpred(X):
                    return dwm_vote(BE.scores(MB, X), OE.scores(X), lwB, lwO)
                rv, rh = macro_f1(yv, mixpred(Xv)), macro_f1(yhist, mixpred(Xhist))
            elif is_H:
                def mixpred(X):
                    return (w * BE.scores(MB, X) + (1 - w) * OE.scores(X)).argmax(1)
                rv, rh = macro_f1(yv, mixpred(Xv)), macro_f1(yhist, mixpred(Xhist))
            else:
                rv, rh = BE.f1(MB, Xv, yv), BE.f1(MB, Xhist, yhist)
            rec.update(ret_val=rv, ret_hist=rh, accepted=accepted, attempted=attempted)
            rows.append(rec)
            gc.collect()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", default="cadence_core.py")
    ap.add_argument("--runner", default="cadence_runner.py")
    ap.add_argument("--temporal", default="cadence_temporal.py")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--datasets", nargs="+",
                    default=["TON_IoT", "CICIDS2017", "Edge_IIoTset",
                             "UNSW_NB15", "IoT23", "MQTTset"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS)
    ap.add_argument("--rounds", type=int, default=160)
    ap.add_argument("--buffer-per-class", type=int, default=200)
    ap.add_argument("--hist-frac", type=float, default=0.20)
    ap.add_argument("--substages", type=int, default=1,
                    help="chunks per stage (finer adaptation granularity)")
    ap.add_argument("--gate", choices=["prequential", "guard"], default="prequential",
                    help="signal that sets the trust weight (see run_method)")
    ap.add_argument("--beta", type=float, default=30.0,
                    help="gate sharpness: w = softmax(beta*[F1_B, F1_O])[0]")
    ap.add_argument("--w-ema", type=float, default=0.7,
                    help="weight on the NEW gate value (1.0 = no smoothing)")
    ap.add_argument("--beta-dwm", type=float, default=0.5,
                    help="DWM penalty per wrong record (Kolter & Maloof default 0.5)")
    ap.add_argument("--arf-models", type=int, default=10)
    ap.add_argument("--warm", type=int, default=4000,
                    help="init samples used to warm-start the online experts")
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--order", choices=["file", "time"], default="file",
                    help="stream order: released-file order (default) or timestamp order (rerun 2)")
    ap.add_argument("--time-col", default=None, help="timestamp column name (auto-detected if omitted)")
    ap.add_argument("--out", default="hybrid_results.csv")
    a = ap.parse_args()

    from sklearn.model_selection import train_test_split as tts
    T = load(a.temporal, "cadence_temporal")
    C = T.load_core(a.core); R = T.load_runner(a.runner)
    C.DS_CONFIGS.setdefault("CICIDS2017_chrono", dict(CHRONO_CONFIG))
    R.remap_paths(C, a.data_root)

    rows = []
    for ds in a.datasets:
        if ds not in C.DS_CONFIGS:
            print("skip", ds); continue
        for seed in a.seeds:
            print("\n==== %s seed=%d  (gate=%s, beta=%.0f, substages=%d) ====" % (ds, seed, a.gate, a.beta, a.substages), flush=True)
            if a.budget:
                C.DS_CONFIGS[ds]["n_samples"] = a.budget
            try:
                Xtr_full, Xv, ytr_full, yv, X_str, y_str, NC = T.load_temporal(C, R, ds, seed, order=a.order, time_col=a.time_col)
            except ValueError as e:
                print("    SKIP:", e, flush=True); continue
            mn = np.bincount(ytr_full, minlength=NC).min()
            try:
                Xtr, Xhist, ytr, yhist = tts(Xtr_full, ytr_full, test_size=a.hist_frac,
                                             random_state=seed,
                                             stratify=ytr_full if mn >= 2 else None)
            except Exception:
                Xtr, Xhist, ytr, yhist = tts(Xtr_full, ytr_full, test_size=a.hist_frac,
                                             random_state=seed)
            print("    train=%d val=%d hist=%d stream=%d classes=%d"
                  % (len(ytr), len(yv), len(yhist), len(y_str), NC), flush=True)
            for m in a.methods:
                t0 = time.time()
                try:
                    rr = run_method(C, m, seed, a, Xtr, ytr, Xv, yv, Xhist, yhist,
                                    X_str, y_str, NC)
                    for r in rr:
                        r.update(dataset=ds, seed=seed, method=m, order=a.order)
                        rows.append(r)
                    pd.DataFrame(rows).to_csv(a.out, index=False)
                    d = pd.DataFrame(rr)
                    msg = "    %-12s CL-F1=%.4f  ret_hist=%.4f" % (m, d.f1.mean(), d.ret_hist.mean())
                    if m in ("CADENCE-H", "CADENCE-H-ng", "DWM", "CADENCE-H-srp", "DWM-srp"):
                        msg += "  |  F_B=%.4f F_O=%.4f  w=%.2f  fix50=%.4f hard=%.4f disagree=%.3f union=%.3f" % (
                            d.f1_B.mean(), d.f1_O.mean(), d.w.mean(), d.f1_fix50.mean(), d.f1_hard.mean(),
                            d.disagree.mean(), d.oracle_union.mean())
                        best = max(FIXW, key=lambda fw: d["f1_fix%02d" % int(round(fw * 10))].mean())
                        msg += "  bestfix=w%.1f:%.4f" % (best, d["f1_fix%02d" % int(round(best * 10))].mean())
                    if d.accepted.notna().any():
                        msg += "  acc=%.0f%%" % (100 * d.accepted.dropna().astype(float).mean())
                    print(msg + "  (%.0fs)" % (time.time() - t0), flush=True)
                except Exception as e:
                    import traceback
                    print("    %-12s FAILED: %r" % (m, e), flush=True); traceback.print_exc()
            del Xtr_full, Xtr, Xhist, Xv, X_str; gc.collect()

    df = pd.DataFrame(rows); df.to_csv(a.out, index=False)
    print("\n[saved]", a.out)
    if len(df):
        DR = ["CICIDS2017", "UNSW_NB15", "IoT23", "MQTTset"]
        per = df.groupby(["method", "dataset", "seed"])[["f1", "ret_hist"]].mean().reset_index()
        print("\n===== drift subset: CL-F1 / independent retention =====")
        print(per[per.dataset.isin(DR)].groupby("method")[["f1", "ret_hist"]].mean()
                .sort_values("f1", ascending=False).round(4).to_string())
        print("\n===== all six streams =====")
        print(per.groupby("method")[["f1", "ret_hist"]].mean()
                .sort_values("f1", ascending=False).round(4).to_string())
        print("\n===== per-dataset CL-F1 =====")
        print(per.pivot_table(index="dataset", columns="method", values="f1").round(4).to_string())


if __name__ == "__main__":
    main()
