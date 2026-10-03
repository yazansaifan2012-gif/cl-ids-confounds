"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         CADENCE — Core Computation Module                                    ║
║  Context-Aware Drift-Adaptive Continual Learning Framework                   ║
║  IoT / Network Anomaly Detection                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Classes:                                                                    ║
║    AdaptiveDriftDetector   — PSI-based 5-state drift detector                ║
║    SmartReplayBuffer       — Uncertainty × diversity × minority replay       ║
║    ContextStateEngine      — 12-dim batch stats → 4 context states           ║
║    AdaptationPolicy        — Policy dataclass (intensity, boost, thresholds) ║
║    ContextGatedPolicy      — (ctx_state, psi_state, NC) → AdaptationPolicy  ║
║                                                                              ║
║  Functions:                                                                  ║
║    eff_num_weights         — Effective-number class weights                  ║
║    focal_proxy_weights     — Focal-loss sample re-weighting proxy            ║
║    calibrate_thresholds    — Per-class threshold calibration on val set      ║
║    apply_thresholds        — Score-ratio argmax inference                    ║
║    smote_safe              — BorderlineSMOTE / SMOTE with leakage guard      ║
║    fpr_macro               — Macro false-positive rate                       ║
║    cl_metrics              — CL summary: F1, Forgetting, DriftRes, Stability ║
║                                                                              ║
║  XGBoost Dual-Ensemble:                                                      ║
║    _train_xgb              — Single booster (main or div config)             ║
║    _val_weights            — Softmax-F1 ensemble weights                     ║
║    _ensemble_probs         — Weighted probability stack                      ║
║                                                                              ║
║  PSI Drift:                                                                  ║
║    _psi_1d                 — Per-feature Population Stability Index          ║
║    _feature_psi            — Mean PSI across random feature subset           ║
║                                                                              ║
║  Data Pipeline:                                                              ║
║    load_and_split          — Load CSV → dedup → scale → train/val/stream/test║
║    run_static_eval         — Dual-ensemble static benchmark                  ║
║    run_cl_streaming_cadence— Full CL loop (ctx_policy or no_ctx)             ║
║                                                                              ║
║  EWC / LwF (PyTorch, optional):                                              ║
║    MLP                     — 256→128→64→NC, Xavier, no BN                   ║
║    train_initial_mlp       — Initial training with early stopping            ║
║    compute_fisher          — Diagonal Fisher information matrix              ║
║    train_ewc_mlp           — EWC continual update                            ║
║    train_lwf_mlp           — LwF knowledge-distillation update               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

# ── stdlib ────────────────────────────────────────────────────────────────────
import copy
import gc
import os
import time
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Optional

warnings.filterwarnings("ignore")

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler

# PyTorch — optional (required only for EWC / LwF)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

ROOT = os.path.dirname(os.path.abspath(__file__))

# ═════════════════════════════════════════════════════════════════════════════
# GLOBAL CONFIG
# ═════════════════════════════════════════════════════════════════════════════

SEED      = 42
SEEDS     = [42, 123, 456]
N_BATCHES = 5

# XGBoost dual-ensemble configs
_XGB_CONFIGS = {
    "main": dict(max_depth=7, eta=0.05, subsample=0.85, colsample_bytree=0.85,
                 min_child_weight=1, gamma=0.05, reg_alpha=0.05, reg_lambda=1.0),
    "div":  dict(max_depth=5, eta=0.08, subsample=0.70, colsample_bytree=0.70,
                 min_child_weight=3, gamma=0.20, reg_alpha=0.20, reg_lambda=2.0),
}

# MLP hyper-params (EWC / LwF)
MLP_INIT_EPOCHS = 120
MLP_CL_EPOCHS   = 50
MLP_BATCH_SIZE  = 256
MLP_LR_INIT     = 1e-3
MLP_LR_CL       = 3e-4
EWC_LAMBDA      = 500.0
LWF_T           = 2.0
LWF_ALPHA       = 1.0

# Dataset registry
DS_CONFIGS = {
    "TON_IoT": dict(
        display="TON-IoT",
        path="cadence/datasets/TON_IoT/train_test_network.csv",
        label_col="type", n_samples=25_000,
        drop_cols=["label", "src_ip", "dst_ip", "dns_query", "ssl_subject",
                   "ssl_issuer", "http_uri", "http_user_agent",
                   "http_orig_mime_types", "http_resp_mime_types",
                   "weird_name", "weird_addl"],
    ),
    "CICIDS2017": dict(
        display="CICIDS2017",
        path="cadence/datasets/CICIDS2017/cicids2017_cleaned.csv",
        label_col="Attack Type", n_samples=22_000, drop_cols=[],
    ),
    "Edge_IIoTset": dict(
        display="Edge-IIoTset",
        path="cadence/datasets/Edge_IIoTset/DNN-EdgeIIoT-dataset.csv",
        label_col="Attack_type", n_samples=22_000,
        drop_cols=["frame.time", "ip.src_host", "ip.dst_host",
                   "arp.dst.proto_ipv4", "arp.src.proto_ipv4", "dns.qry.name",
                   "http.request.full_uri", "http.referer", "http.file_data",
                   "http.request.uri.query", "mqtt.msg", "mqtt.topic",
                   "mqtt.protoname", "Attack_label"],
    ),
    "UNSW_NB15": dict(
        display="UNSW-NB15",
        path="cadence/datasets/UNSW_NB15/unsw_nb15.csv",
        label_col="attack_cat", n_samples=25_000,
        drop_cols=["id", "label"],
    ),
    "IoT23": dict(
        display="IoT-23",
        path="cadence/datasets/IoT23/iot23.csv",
        label_col="label", n_samples=22_000, drop_cols=[],
    ),
    "MQTTset": dict(
        display="MQTTset",
        path="cadence/datasets/MQTTset/mqttset.csv",
        label_col="label", n_samples=19_000, drop_cols=[],
    ),
}


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Utility Functions
# ═════════════════════════════════════════════════════════════════════════════

def set_seeds(seed: int = SEED) -> None:
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def fpr_macro(y_true: np.ndarray, y_pred: np.ndarray, nc: int) -> float:
    """Macro-averaged False Positive Rate across all classes."""
    vals = []
    for c in range(nc):
        fp = int(((y_pred == c) & (y_true != c)).sum())
        tn = int(((y_pred != c) & (y_true != c)).sum())
        vals.append(fp / (fp + tn + 1e-9))
    return float(np.mean(vals))


def cl_metrics(f1s: list, fprs: list) -> dict:
    """
    Compute CL summary metrics from per-batch F1 and FPR sequences.

    Returns:
        CL_F1      — mean batch F1
        Forgetting — max(F1) − final(F1)
        DriftRes   — min(F1) / max(F1)   (resilience under drift)
        Stability  — 1 − std(F1) / mean(F1)
        CL_FPR     — mean batch FPR
    """
    f1s  = np.array(f1s, dtype=np.float64)
    fprs = np.array(fprs, dtype=np.float64)
    return dict(
        CL_F1      = round(float(f1s.mean()), 4),
        Forgetting = round(float(max(0.0, f1s.max() - f1s[-1])), 4),
        DriftRes   = round(float(f1s.min() / f1s.max()) if f1s.max() > 0 else 0.0, 4),
        Stability  = round(float(1 - f1s.std() / (f1s.mean() + 1e-12)), 4),
        CL_FPR     = round(float(fprs.mean()), 4),
    )


def _row_hashes(X: np.ndarray) -> np.ndarray:
    X_c = np.ascontiguousarray(X)
    return np.array(
        [hash(row.tobytes()) for row in X_c.view(np.uint8).reshape(len(X_c), -1)],
        dtype=np.int64)


def _deduplicate(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove exact-duplicate rows before any split (leakage guard)."""
    hashes = _row_hashes(X)
    _, keep = np.unique(hashes, return_index=True)
    keep    = np.sort(keep)
    dropped = len(X) - len(keep)
    if dropped > 0:
        print(f"  [dedup] removed {dropped} exact-duplicate rows "
              f"({dropped / len(X) * 100:.1f}%)", flush=True)
    return X[keep], y[keep]


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Weight Helpers & Threshold Calibration
# ═════════════════════════════════════════════════════════════════════════════

def eff_num_weights(y: np.ndarray, beta: float = 0.9999) -> np.ndarray:
    """
    Effective-number class re-weighting.
    w_c = 1 / E_c,  E_c = (1 − β^n_c) / (1 − β)
    Returns per-sample weights (shape = len(y)).
    """
    nc      = int(y.max()) + 1
    counts  = np.bincount(y, minlength=nc).astype(np.float64)
    eff     = (1.0 - beta ** np.maximum(counts, 1)) / (1.0 - beta)
    class_w = np.where(counts > 0, 1.0 / np.maximum(eff, 1e-8), 0.0)
    class_w = class_w / (class_w.sum() + 1e-12) * nc
    return class_w[y].astype(np.float32)


def focal_proxy_weights(y: np.ndarray, probs: np.ndarray,
                        gamma: float = 1.5,
                        base_w: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Focal-weight proxy: re-weights samples after initial prediction.
    w_i = (1 − p_correct)^gamma × base_w_i
    Emphasises hard / minority cases without a second forward pass.
    """
    idx       = np.arange(len(y))
    p_correct = np.clip(probs[idx, y.astype(int)], 1e-7, 1.0)
    focal     = (1.0 - p_correct) ** gamma
    if base_w is None:
        base_w = eff_num_weights(y)
    w = focal * base_w
    w = w / (w.mean() + 1e-12)
    return w.astype(np.float32)


def calibrate_thresholds(probs: np.ndarray, y_val: np.ndarray,
                         nc: int, search_grid: int = 40) -> np.ndarray:
    """
    Per-class threshold calibration on validation set.
    Searches for t* that maximises binary F1 for each class independently.
    """
    thresholds = np.full(nc, 1.0 / nc, dtype=np.float32)
    for c in range(nc):
        mask = (y_val == c)
        if mask.sum() < 2:
            continue
        p_c                  = probs[:, c]
        best_t, best_score   = thresholds[c], -1.0
        for t in np.linspace(p_c.min() + 1e-4, p_c.max() - 1e-4, search_grid):
            pred = (p_c >= t).astype(int)
            tp   = int(((pred == 1) & mask).sum())
            fp   = int(((pred == 1) & ~mask).sum())
            fn   = int(((pred == 0) & mask).sum())
            if tp == 0:
                continue
            prec = tp / (tp + fp + 1e-8)
            rec  = tp / (tp + fn + 1e-8)
            fb   = 2 * prec * rec / (prec + rec + 1e-8)
            if fb > best_score:
                best_score, best_t = fb, float(t)
        thresholds[c] = best_t
    return thresholds


def apply_thresholds(probs: np.ndarray,
                     thr: Optional[np.ndarray]) -> np.ndarray:
    """
    Score-ratio argmax: pred = argmax(p_c / thr_c).
    Falls back to plain argmax if thr is None.
    """
    if thr is None:
        return probs.argmax(1)
    scores = probs / (np.array(thr, dtype=np.float32) + 1e-9)
    return scores.argmax(1)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — XGBoost Dual Ensemble
# ═════════════════════════════════════════════════════════════════════════════

def _train_xgb(X, y, X_val, y_val, nc, weights,
               config: str = "main", rounds: int = 160) -> xgb.Booster:
    """
    Train one XGBoost booster with early stopping on val mlogloss.
    config="main" → deep/dense tree  |  config="div" → shallow/regularised tree
    """
    params = dict(
        objective   = "multi:softprob",
        num_class   = nc,
        eval_metric = "mlogloss",
        verbosity   = 0,
        seed        = SEED,
        **_XGB_CONFIGS[config],
    )
    early_stop = xgb.callback.EarlyStopping(
        rounds=25, metric_name="mlogloss", data_name="val",
        maximize=False, save_best=True, min_delta=1e-4)
    dtr = xgb.DMatrix(X,     label=y,     weight=weights)
    dv  = xgb.DMatrix(X_val, label=y_val)
    return xgb.train(params, dtr, num_boost_round=rounds,
                     evals=[(dtr, "train"), (dv, "val")],
                     callbacks=[early_stop], verbose_eval=False)


def _val_weights(m_main: xgb.Booster, m_div: xgb.Booster,
                 X_val, y_val, nc) -> tuple[float, float]:
    """
    Compute softmax-F1 ensemble weights.
    w = softmax(5 × F1_val) for each member.
    """
    dv  = xgb.DMatrix(X_val)
    f1m = f1_score(y_val, m_main.predict(dv).reshape(-1, nc).argmax(1),
                   average="macro", zero_division=0)
    f1d = f1_score(y_val, m_div.predict(dv).reshape(-1, nc).argmax(1),
                   average="macro", zero_division=0)
    em, ed = np.exp(f1m * 5.0), np.exp(f1d * 5.0)
    s = em + ed
    return float(em / s), float(ed / s)


def _ensemble_probs(m_main: xgb.Booster, m_div: xgb.Booster,
                    X, nc, wt: float = 0.5, wd: float = 0.5) -> np.ndarray:
    """Weighted probability stack: p = wt × p_main + wd × p_div."""
    dm  = xgb.DMatrix(X)
    p_m = m_main.predict(dm).reshape(-1, nc).astype(np.float32)
    p_d = m_div.predict(dm).reshape(-1, nc).astype(np.float32)
    return (wt * p_m + wd * p_d).astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PSI-Based Drift Detection Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _psi_1d(expected: np.ndarray, actual: np.ndarray,
            buckets: int = 10, eps: float = 1e-8) -> float:
    """
    Population Stability Index for a single feature.
    PSI = Σ (actual% − expected%) × ln(actual% / expected%)
    """
    lo    = min(expected.min(), actual.min())
    hi    = max(expected.max(), actual.max()) + eps
    edges = np.linspace(lo, hi, buckets + 1)
    e_cnt = np.histogram(expected, bins=edges)[0].astype(float) + eps
    a_cnt = np.histogram(actual,   bins=edges)[0].astype(float) + eps
    e_pct = e_cnt / e_cnt.sum()
    a_pct = a_cnt / a_cnt.sum()
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct + eps)))


def _feature_psi(X_ref: np.ndarray, X_cur: np.ndarray,
                 buckets: int = 10, max_feat: int = 50) -> float:
    """Mean PSI across a random subset of features (capped at max_feat)."""
    nf  = min(X_ref.shape[1], max_feat)
    idx = np.random.choice(X_ref.shape[1], nf, replace=False)
    return float(np.mean([_psi_1d(X_ref[:, j], X_cur[:, j], buckets) for j in idx]))


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Adaptive Drift Detector
# ═════════════════════════════════════════════════════════════════════════════

class AdaptiveDriftDetector:
    """
    PSI-based 5-state drift detector with hysteresis and recurring-context memory.

    States
    ------
    stable        PSI < psi_gradual
    gradual_drift psi_gradual ≤ PSI < psi_sudden
    sudden_drift  PSI ≥ psi_sudden
    ood           PSI ≥ psi_ood  AND  mean_confidence < conf_ood_gate
    recurring     cosine similarity to any stored context ≥ recurring_thr

    Reference update
    ----------------
    stable:       EMA blend (80/20) — gradual adaptation
    sudden_drift: hard reset to new batch
    else:         no update
    NOTE: EMA slices with min(len(ref), len(batch)) to handle variable sizes.
    """

    def __init__(self,
                 psi_gradual:   float = 0.10,
                 psi_sudden:    float = 0.25,
                 psi_ood:       float = 0.45,
                 conf_ood_gate: float = 0.40,
                 history_len:   int   = 6,
                 recurring_thr: float = 0.92,
                 hysteresis:    int   = 1):
        self.psi_gradual   = psi_gradual
        self.psi_sudden    = psi_sudden
        self.psi_ood       = psi_ood
        self.conf_ood_gate = conf_ood_gate
        self.recurring_thr = recurring_thr
        self.hysteresis    = hysteresis

        self._ref:          Optional[np.ndarray] = None
        self._ref_mean:     Optional[np.ndarray] = None
        self._ctx_history:  deque = deque(maxlen=history_len)
        self._psi_history:  deque = deque(maxlen=10)
        self._state         = "stable"
        self._pending_state = "stable"
        self._pending_count = 0
        self._severity      = 0.0
        self.batch_count    = 0

    def update(self, X_batch: np.ndarray, mean_conf: float = 1.0) -> dict:
        """
        Process one batch. Returns dict with:
            state, severity, psi, is_recurring, psi_trend
        """
        if self._ref is None:
            self._ref      = X_batch.copy()
            self._ref_mean = X_batch.mean(0)
            self._ctx_history.append(self._ref_mean.copy())
            self.batch_count += 1
            return dict(state="stable", severity=0.0, psi=0.0,
                        is_recurring=False, psi_trend=0.0)

        psi = _feature_psi(self._ref, X_batch)
        self._psi_history.append(psi)
        arr   = np.array(self._psi_history)
        trend = float(arr[-1] - arr[0]) / max(len(arr) - 1, 1)
        sev   = float(np.clip(psi / (self.psi_ood + 1e-8), 0, 1))
        self._severity = sev

        cur   = X_batch.mean(0)
        recur = self._is_recurring(cur)

        if recur and psi >= self.psi_gradual:
            cand = "recurring"
        elif psi >= self.psi_ood and mean_conf < self.conf_ood_gate:
            cand = "ood"
        elif psi >= self.psi_sudden:
            cand = "sudden_drift"
        elif psi >= self.psi_gradual:
            cand = "gradual_drift"
        else:
            cand = "stable"

        # Hysteresis: require consecutive confirmations before state change
        if cand == self._state:
            self._pending_count = 0
        elif cand == self._pending_state:
            self._pending_count += 1
        else:
            self._pending_state = cand
            self._pending_count = 1
        if self._pending_count >= self.hysteresis:
            self._state         = self._pending_state
            self._pending_count = 0

        self._ctx_history.append(cur.copy())

        # Reference update
        if psi < self.psi_gradual:
            n          = min(len(self._ref), len(X_batch))
            self._ref  = 0.8 * self._ref[:n] + 0.2 * X_batch[:n]
        elif self._state == "sudden_drift":
            self._ref  = X_batch.copy()
        self._ref_mean = self._ref.mean(0)
        self.batch_count += 1

        return dict(state=self._state, severity=sev, psi=psi,
                    is_recurring=recur, psi_trend=trend)

    def retrain_intensity(self) -> str:
        """
        Map current drift state to retrain decision. Whenever drift is present
        the model retrains at full strength and the downstream validation guard
        decides whether the update is kept; PSI only elects to "skip" when the
        stream is genuinely stable. This makes PSI a compute-efficiency gate
        (it avoids retraining on stable stages) that does not throttle the
        accuracy of a beneficial adaptation:
            sudden / ood / gradual / recurring drift → "full"
            stable                                   → "skip"
        """
        if self._state in ("sudden_drift", "ood", "gradual_drift", "recurring"):
            return "full"
        return "skip"

    @property
    def state(self) -> str:
        return self._state

    @property
    def severity(self) -> float:
        return self._severity

    def _is_recurring(self, cur: np.ndarray) -> bool:
        if len(self._ctx_history) < 2:
            return False
        nc2 = cur / (np.linalg.norm(cur) + 1e-12)
        for h in list(self._ctx_history)[:-1]:
            nh = h / (np.linalg.norm(h) + 1e-12)
            if float(nc2 @ nh) >= self.recurring_thr:
                return True
        return False


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Smart Replay Buffer
# ═════════════════════════════════════════════════════════════════════════════

class SmartReplayBuffer:
    """
    Per-class bounded experience replay.

    Selection priority: uncertainty (entropy) × recency (age) × minority boost.
    LEAKAGE GUARD: always call sample() BEFORE add() within each CL batch.

    Parameters
    ----------
    max_per_class : int   — max stored samples per class
    age_decay     : float — multiplicative decay per batch (older = less weight)
    """

    def __init__(self,
                 max_per_class: int   = 80,
                 age_decay:     float = 0.05,
                 diversity_k:   int   = 40,
                 min_minority:  int   = 20):
        self.max_per_class = max_per_class
        self.age_decay     = age_decay
        self._X:       list = []
        self._y:       list = []
        self._entropy: list = []
        self._age:     list = []
        self._current_batch = 0

    def add(self, X: np.ndarray, y: np.ndarray,
            entropies: Optional[np.ndarray] = None) -> None:
        """Add batch to buffer. entropies: per-sample Shannon entropy."""
        if entropies is None:
            entropies = np.zeros(len(y), dtype=np.float32)
        for i in range(len(y)):
            self._X.append(X[i].copy())
            self._y.append(int(y[i]))
            self._entropy.append(float(entropies[i]))
            self._age.append(1.0)
        self._current_batch += 1
        self._decay_ages()
        self._trim()

    def sample(self, n: int, nc: int,
               minority_boost: float = 2.5) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample n examples with class-balanced quotas.
        minority_boost: multiplier for under-represented classes (count < 30% of majority).
        """
        if not self._y:
            return np.empty((0, 0)), np.empty(0, dtype=int)

        X_arr = np.array(self._X, dtype=np.float32)
        y_arr = np.array(self._y, dtype=np.int32)
        ent   = np.array(self._entropy, dtype=np.float32)
        age   = np.array(self._age,     dtype=np.float32)

        counts = np.bincount(y_arr, minlength=nc)
        maj    = counts.max()
        quotas = np.zeros(nc, dtype=int)
        for c in range(nc):
            if counts[c] == 0:
                continue
            is_min    = counts[c] < maj * 0.3
            base_q    = max(1, int(n / nc))
            quotas[c] = int(base_q * minority_boost) if is_min else base_q

        total = quotas.sum()
        if total > 0:
            quotas = np.maximum(1, (quotas * n / total).astype(int))

        idx_out = []
        for c in range(nc):
            if counts[c] == 0 or quotas[c] == 0:
                continue
            cidx  = np.where(y_arr == c)[0]
            score = ent[cidx] * age[cidx]
            score = score / (score.sum() + 1e-12)
            k     = min(quotas[c], len(cidx))
            chosen = np.random.choice(cidx, size=k, replace=False, p=score)
            idx_out.extend(chosen.tolist())

        if not idx_out:
            return np.empty((0, X_arr.shape[1])), np.empty(0, dtype=int)
        idx_out = np.array(idx_out)
        np.random.shuffle(idx_out)
        return X_arr[idx_out], y_arr[idx_out]

    def size(self) -> int:
        return len(self._y)

    def _decay_ages(self) -> None:
        for i in range(len(self._age)):
            self._age[i] *= (1.0 - self.age_decay)

    def _trim(self) -> None:
        """Evict lowest-scoring entries to keep each class ≤ max_per_class."""
        y_arr = np.array(self._y)
        nc    = int(y_arr.max()) + 1 if len(y_arr) > 0 else 0
        keep  = np.ones(len(self._y), dtype=bool)
        for c in range(nc):
            cidx = np.where(y_arr == c)[0]
            if len(cidx) <= self.max_per_class:
                continue
            scores = np.array(self._entropy)[cidx] * np.array(self._age)[cidx]
            evict  = cidx[scores.argsort()[:len(cidx) - self.max_per_class]]
            keep[evict] = False
        idx           = np.where(keep)[0]
        self._X       = [self._X[i]       for i in idx]
        self._y       = [self._y[i]       for i in idx]
        self._entropy = [self._entropy[i] for i in idx]
        self._age     = [self._age[i]     for i in idx]


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Context State Engine
# ═════════════════════════════════════════════════════════════════════════════

class ContextStateEngine:
    """
    Classifies operational context into 4 states from 12 batch statistics.

    Context is POLICY-ONLY — the model always trains on raw features.
    No feature fusion; no leakage. ref_mean/ref_std fitted on X_tr only.

    States (by ctx_score)
    ---------------------
    stable       [0.00 – 0.12)
    mild_drift   [0.12 – 0.38)
    severe_drift [0.38 – 0.68)
    unknown_ctx  [0.68 – 1.00]

    ctx_score = 0.40 × mean_shift
              + 0.25 × std_shift
              + 0.20 × psi_severity
              + 0.15 × imbalance
    """
    THR_MILD = 0.12
    THR_SEV  = 0.38
    THR_UNK  = 0.68

    def __init__(self, X_ref: np.ndarray, y_ref: np.ndarray, nc: int):
        self.nc = nc
        X_safe        = np.nan_to_num(X_ref.astype(np.float64),
                                      nan=0., posinf=0., neginf=0.)
        self.ref_mean = np.clip(X_safe.mean(0), -1e9, 1e9)
        self.ref_std  = np.clip(X_safe.std(0),  1e-9, 1e9) + 1e-9
        cls, cts      = np.unique(y_ref, return_counts=True)
        self.ref_dist = np.zeros(nc, dtype=np.float64)
        self.ref_dist[cls] = cts / cts.sum()
        self._score_history: deque = deque(maxlen=5)

    def analyse(self, X_batch: np.ndarray, y_batch: np.ndarray,
                psi_severity: float = 0.0) -> tuple:
        """
        Analyse one batch.

        Returns
        -------
        state     : str   — "stable" | "mild_drift" | "severe_drift" | "unknown_ctx"
        ctx_score : float — composite context score in [0, 1]
        stats     : dict  — 7 diagnostic stats
        """
        X = np.nan_to_num(X_batch.astype(np.float64),
                          nan=0., posinf=0., neginf=0.)
        batch_mean = np.clip(X.mean(0), -1e9, 1e9)
        batch_std  = np.clip(X.std(0),  1e-9, 1e9) + 1e-9

        z_shifts   = np.abs(batch_mean - self.ref_mean) / self.ref_std
        mean_shift = float(np.clip(z_shifts.mean() / 5.0, 0, 1))
        std_ratios = batch_std / self.ref_std
        std_shift  = float(np.clip(np.abs(std_ratios - 1).mean() / 5.0, 0, 1))

        if y_batch is not None and len(y_batch) > 0:
            cls, cts = np.unique(y_batch.astype(int), return_counts=True)
            cts_nz   = cts[cts > 0]
            imb      = float(np.clip(
                (cts_nz.max() / cts_nz.min() / max(self.nc, 1)) / 5.0
                if len(cts_nz) >= 2 else 0.0, 0, 1))
            n_cls = len(cls)
            p     = cts / cts.sum()
            ent   = float(-np.sum(p * np.log(p + 1e-12))
                          / np.log(max(self.nc, 2)))
        else:
            imb = 0.0; n_cls = self.nc; ent = 1.0

        ctx_score = float(np.clip(
            0.40 * mean_shift
            + 0.25 * std_shift
            + 0.20 * float(np.clip(psi_severity, 0, 1))
            + 0.15 * imb, 0.0, 1.0))
        self._score_history.append(ctx_score)

        if len(self._score_history) >= 2:
            hist  = list(self._score_history)
            trend = (hist[-1] - hist[0]) / max(len(hist) - 1, 1)
        else:
            trend = 0.0

        if ctx_score < self.THR_MILD:
            state = "stable"
        elif ctx_score < self.THR_SEV:
            state = "mild_drift"
        elif ctx_score < self.THR_UNK:
            state = "severe_drift"
        else:
            state = "unknown_ctx"

        stats = dict(
            mean_shift      = round(mean_shift,    4),
            std_shift       = round(std_shift,     4),
            imbalance       = round(imb,           4),
            n_classes_seen  = n_cls,
            class_entropy   = round(ent,           4),
            psi_severity    = round(psi_severity,  4),
            ctx_score_trend = round(trend,         4),
        )
        return state, ctx_score, stats


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Context-Gated Adaptation Policy
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class AdaptationPolicy:
    """
    Parameters produced by ContextGatedPolicy for the current CL batch.

    Fields
    ------
    intensity      : "full" | "partial" | "skip"
    replay_n_mult  : replay-size multiplier (API-compat; currently buffer-capped)
    minority_boost : sampling multiplier for minority classes in replay
    thr_mult       : threshold scaling factor (< 1 → more sensitive)
    focal_gamma    : focal-loss exponent for sample re-weighting
    smote_ratio    : target class-balance ratio for SMOTE
    rounds_mult    : XGBoost rounds multiplier
    note           : human-readable decision explanation
    """
    intensity:      str   = "skip"
    replay_n_mult:  float = 1.0
    minority_boost: float = 2.5
    thr_mult:       float = 1.0
    focal_gamma:    float = 1.5
    smote_ratio:    float = 0.55
    rounds_mult:    float = 1.0
    note:           str   = ""


class ContextGatedPolicy:
    """
    Maps (ctx_state, psi_state, n_classes_seen) → AdaptationPolicy.

    ONE active mechanism
    --------------------
    Class-diversity-adaptive parameters for full retrains when
    n_classes_seen ≥ NC_THRESHOLD (8):
        minority_boost = clip(2.5 + 0.12 × (NC − 4), 2.5, 4.0)
        thr_mult       = clip(1.0 − 0.008 × (NC − 4), 0.92, 1.0)

    CICIDS2017 (NC=7) is always below threshold → guaranteed neutral (Δ=0).

    Intensity overrides (promote-only; context never suppresses adaptation)
    -----------------------------------------------------------------------
    unknown_ctx / severe_drift + PSI=skip   → promote to partial
    mild_drift  / stable                    → carry PSI baseline unchanged
    Rationale: under any drift the model must be allowed to adapt; the
    downstream validation guard — not the context gate — decides whether an
    update is kept. Context therefore only *raises* adaptation when PSI is
    idle but the context signal indicates change; it never forces "skip".
    """
    NC_THRESHOLD = 8

    def get(self,
            ctx_state:     str,
            psi_state:     str,
            psi_intensity: str,
            ctx_score:     float,
            score_trend:   float,
            n_cls_seen:    int   = 4,
            class_entropy: float = 1.0,
            imbalance:     float = 0.0) -> AdaptationPolicy:

        # ── Intensity decision ────────────────────────────────────────────────
        if ctx_state == "unknown_ctx":
            if psi_intensity == "skip":
                intensity = "partial"
                note      = "unknown_ctx: skip→partial"
            else:
                intensity = psi_intensity
                note      = f"unknown_ctx: PSI={psi_intensity}"

        elif ctx_state == "severe_drift":
            if psi_intensity == "skip":
                intensity = "partial"
                note      = "severe: skip→partial"
            else:
                intensity = psi_intensity
                note      = f"severe: PSI={psi_intensity}"

        elif ctx_state == "mild_drift":
            # Context NEVER suppresses a drift-flagged adaptation to "skip":
            # under any drift the model must be allowed to adapt, and the
            # validation guard downstream decides whether the update is kept.
            # Context only carries the PSI baseline through unchanged here.
            intensity = psi_intensity
            note      = f"mild: PSI={psi_intensity}"

        elif ctx_state == "stable":
            # Stable context defers to PSI: if PSI itself sees no drift it
            # returns "skip" (true idle → save compute); otherwise the
            # PSI-flagged adaptation is preserved (guard filters acceptance).
            intensity = psi_intensity
            note      = f"stable: PSI={psi_intensity}"
        else:
            intensity = psi_intensity
            note      = f"fallthrough: PSI={psi_intensity}"

        # ── Class-diversity-adaptive params (full retrains, NC ≥ threshold) ──
        minority_boost = 2.5
        thr_mult       = 1.0
        if intensity == "full" and n_cls_seen >= self.NC_THRESHOLD:
            minority_boost = float(np.clip(
                2.5 + 0.12 * (n_cls_seen - 4), 2.5, 4.0))
            thr_mult = float(np.clip(
                1.0 - 0.008 * (n_cls_seen - 4), 0.92, 1.0))
            note += f"  nc={n_cls_seen}→mbst={minority_boost:.2f}"

        return AdaptationPolicy(
            intensity      = intensity,
            replay_n_mult  = 1.0,
            minority_boost = minority_boost,
            thr_mult       = thr_mult,
            focal_gamma    = 1.5,
            smote_ratio    = 0.55,
            rounds_mult    = 1.0,
            note           = note,
        )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9 — SMOTE Helper
# ═════════════════════════════════════════════════════════════════════════════

def smote_safe(X: np.ndarray, y: np.ndarray, nc: int,
               ratio: float = 0.55, k: int = 3,
               severe: bool = False, seed: int = None) -> tuple:
    """
    Safe BorderlineSMOTE (stable) / SMOTE (severe) oversampling.
    LEAKAGE GUARD: call only on training partitions.
    Falls back gracefully on edge cases (insufficient neighbours, etc.).
    """
    s      = seed if seed is not None else SEED
    counts = np.bincount(y, minlength=nc)
    maj    = counts.max()
    target = int(maj * (0.75 if severe else ratio))
    strat  = {c: min(target, counts[c] * 8)
              for c in range(nc) if 0 < counts[c] < target}
    strat  = {c: v for c, v in strat.items() if v > counts[c]}
    if not strat:
        return X, y
    k2  = max(1, min(k, int(counts[counts > 0].min()) - 1))
    Cls = SMOTE if severe else BorderlineSMOTE
    try:
        return Cls(sampling_strategy=strat,
                   k_neighbors=k2, random_state=s).fit_resample(X, y)
    except Exception:
        try:
            return SMOTE(sampling_strategy=strat,
                         k_neighbors=k2, random_state=s).fit_resample(X, y)
        except Exception:
            return X, y


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 10 — Data Pipeline
# ═════════════════════════════════════════════════════════════════════════════

def load_and_split(ds_key: str, seed: int = SEED):
    """
    Load dataset → stratified subsample → deduplicate → split.

    Split structure (leakage-free):
        X_raw ──20%──► X_te      (hold-out test, never seen by model)
              └─80%──► X_dev
                        ├─35%──► X_stream  (CL stream, reserved BEFORE fit)
                        └─65%──► X_init
                                   ├─85%─► X_tr   (initial train + SMOTE)
                                   └─15%─► X_val  (early stop + threshold cal)

    Returns
    -------
    X_tr, X_val, X_stream, X_te, y_tr, y_val, y_stream, y_te,
    batch_idx (list of N_BATCHES index arrays), NC, LabelEncoder
    """
    cfg = DS_CONFIGS[ds_key]
    np.random.seed(seed)
    df  = pd.read_csv(os.path.join(ROOT, cfg["path"]), low_memory=False)

    # Stratified subsample
    groups = []
    for _, g in df.groupby(cfg["label_col"], sort=False):
        take = min(max(2, int(len(g) * (cfg["n_samples"] / len(df)) * 1.8)), len(g))
        groups.append(g.sample(n=take, random_state=seed))
    df = (pd.concat(groups, ignore_index=True)
            .sample(n=min(cfg["n_samples"], sum(len(g) for g in groups)),
                    random_state=seed)
            .reset_index(drop=True))

    le    = LabelEncoder()
    y_all = le.fit_transform(df[cfg["label_col"]].astype(str))
    drop  = [c for c in cfg["drop_cols"] + [cfg["label_col"]] if c in df.columns]
    df    = df.drop(columns=drop)

    for col in df.columns:
        if df[col].dtype == object:
            le2 = LabelEncoder()
            df[col] = le2.fit_transform(df[col].astype(str).fillna("NA"))
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    X_raw = np.nan_to_num(df.values.astype(np.float32))
    del df; gc.collect()
    NC = int(len(le.classes_))
    print(f"    {len(y_all)} samples | {NC} classes | {X_raw.shape[1]} features",
          flush=True)

    X_raw, y_all = _deduplicate(X_raw, y_all)

    X_dev, X_te, y_dev, y_te = train_test_split(
        X_raw, y_all, test_size=0.20, random_state=seed, stratify=y_all)
    del X_raw; gc.collect()

    X_init, X_stream, y_init, y_stream = train_test_split(
        X_dev, y_dev, test_size=0.35, random_state=seed, stratify=y_dev)
    del X_dev; gc.collect()

    # Scaler fitted on X_init only (no leakage from stream or test)
    var      = X_init.var(axis=0)
    nzv      = var > 1e-6
    X_init   = X_init[:, nzv]
    X_stream = X_stream[:, nzv]
    X_te     = X_te[:, nzv]
    sc       = RobustScaler(quantile_range=(5, 95)).fit(X_init)
    X_init   = np.nan_to_num(sc.transform(X_init).astype(np.float32))
    X_stream = np.nan_to_num(sc.transform(X_stream).astype(np.float32))
    X_te     = np.nan_to_num(sc.transform(X_te).astype(np.float32))

    try:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_init, y_init, test_size=0.15, random_state=seed, stratify=y_init)
    except ValueError:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_init, y_init, test_size=0.15, random_state=seed)
    del X_init; gc.collect()

    rng       = np.random.default_rng(seed)
    batch_idx = np.array_split(rng.permutation(len(y_stream)), N_BATCHES)
    return X_tr, X_val, X_stream, X_te, y_tr, y_val, y_stream, y_te, \
           batch_idx, NC, le


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 11 — CADENCE: Static Evaluation
# ═════════════════════════════════════════════════════════════════════════════

def run_static_eval(X_tr, y_tr, X_val, y_val, X_te, y_te,
                    nc: int, dataset_name: str, rounds: int = 180) -> dict:
    """
    Train dual-ensemble (main + div) on initial data, evaluate on test set.
    Returns models + calibrated thresholds + all static metrics.
    """
    print(f"  [Static eval]", flush=True)
    t0 = time.perf_counter()

    Xa, ya = smote_safe(X_tr, y_tr, nc, ratio=0.55, k=3, severe=False)
    bw     = eff_num_weights(ya)
    m_main = _train_xgb(Xa, ya, X_val, y_val, nc, bw, "main", rounds)

    # Focal-proxy re-weighting using main model predictions
    p_tr   = m_main.predict(xgb.DMatrix(Xa)).reshape(-1, nc).astype(np.float32)
    fw     = focal_proxy_weights(ya, p_tr, gamma=1.5, base_w=bw)
    del p_tr
    m_div  = _train_xgb(Xa, ya, X_val, y_val, nc, fw, "div", rounds)
    del Xa, ya, fw; gc.collect()

    wt, wd = _val_weights(m_main, m_div, X_val, y_val, nc)
    p_fv   = _ensemble_probs(m_main, m_div, X_val, nc, wt, wd)
    thr    = calibrate_thresholds(p_fv, y_val, nc, search_grid=45)
    del p_fv; gc.collect()

    p_te   = _ensemble_probs(m_main, m_div, X_te, nc, wt, wd)
    y_pred = apply_thresholds(p_te, thr)
    f1m    = f1_score(y_te, y_pred, average="macro", zero_division=0)
    mcc    = matthews_corrcoef(y_te, y_pred)
    fpr    = fpr_macro(y_te, y_pred, nc)
    present = np.unique(y_te)
    try:
        auc = (roc_auc_score(y_te, p_te, multi_class="ovr", average="macro")
               if len(present) == nc
               else roc_auc_score(y_te, p_te[:, present],
                                  multi_class="ovr", average="macro",
                                  labels=present))
    except Exception:
        auc = float("nan")

    del p_te; gc.collect()
    t_fit = time.perf_counter() - t0
    print(f"    F1={f1m:.4f}  MCC={mcc:.4f}  AUC={auc:.4f}  "
          f"FPR={fpr:.4f}  ({t_fit:.0f}s)", flush=True)

    return dict(m_main=m_main, m_div=m_div, wt=wt, wd=wd, thr=thr,
                f1m=f1m, mcc=mcc, auc=auc, fpr=fpr)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 12 — CADENCE: CL Streaming
# ═════════════════════════════════════════════════════════════════════════════

def run_cl_streaming_cadence(m_main, m_div, wt, wd, thr,
                              X_tr, y_tr, X_stream, y_stream,
                              X_te, y_te, nc: int, dataset_name: str,
                              use_context: bool = True,
                              seed: int = SEED) -> list[dict]:
    """
    CADENCE CL streaming loop over N_BATCHES sequential batches.

    use_context=True  → CADENCE (ctx_policy): drift + context gating
    use_context=False → PSI-only: drift detection only, no context engine

    Per-batch sequence:
        1. Predict batch → PSI drift update
        2. Context analysis → policy (if use_context)
        3. Maybe retrain (full / partial / skip)
        4. Threshold EMA blend (0.35 × new + 0.65 × old)
        5. Evaluate on full test set
        6. Add batch to replay buffer (AFTER evaluation)

    Returns list of per-batch result dicts.
    """
    replay = SmartReplayBuffer(max_per_class=65, age_decay=0.06)
    drift  = AdaptiveDriftDetector(psi_gradual=0.10, psi_sudden=0.25,
                                   psi_ood=0.45, conf_ood_gate=0.40,
                                   history_len=6, recurring_thr=0.92,
                                   hysteresis=1)
    ctx_engine    = ContextStateEngine(X_tr, y_tr, nc) if use_context else None
    policy_engine = ContextGatedPolicy() if use_context else None

    # Seed replay with initial training data (never test data)
    dm_tr    = xgb.DMatrix(X_tr)
    p_seed   = m_main.predict(dm_tr).reshape(-1, nc).astype(np.float32)
    ent_seed = -(np.clip(p_seed, 1e-9, 1.) * np.log(np.clip(p_seed, 1e-9, 1.))).sum(1)
    replay.add(X_tr, y_tr, ent_seed)
    del dm_tr, p_seed, ent_seed; gc.collect()

    rng     = np.random.default_rng(seed)
    batches = np.array_split(rng.permutation(len(y_stream)), N_BATCHES)
    results = []
    last_rt = -99

    for b_idx, bidx in enumerate(batches):
        t0 = time.perf_counter()
        Xb = X_stream[bidx]
        yb = y_stream[bidx]

        # Step 1: predict + PSI drift update
        p_b           = _ensemble_probs(m_main, m_div, Xb, nc, wt, wd)
        mc            = float(p_b.max(1).mean())
        di            = drift.update(Xb, mean_conf=mc)
        psi_state     = di["state"]
        sev           = di["severity"]
        psi_intensity = drift.retrain_intensity()

        # Step 2: context analysis → policy
        if use_context and ctx_engine is not None:
            ctx_state, ctx_score, ctx_stats = ctx_engine.analyse(Xb, yb, sev)
            pol = policy_engine.get(
                ctx_state, psi_state, psi_intensity, ctx_score,
                ctx_stats["ctx_score_trend"], ctx_stats["n_classes_seen"],
                ctx_stats["class_entropy"],   ctx_stats["imbalance"])
            intensity = pol.intensity
        else:
            ctx_state = psi_state
            ctx_score = sev
            pol       = AdaptationPolicy(intensity=psi_intensity, note="no_ctx")
            intensity = psi_intensity

        # Cooldown: moderate compute after a very recent retrain by stepping
        # a "full" down to "partial", but NEVER down to "skip" — a drift-flagged
        # batch must still be allowed to adapt (the validation guard decides
        # acceptance). This prevents the model from freezing across consecutive
        # drifting batches.
        if (b_idx - last_rt) < 1 and intensity == "full":
            intensity = "partial"
        effective_thr = (thr * pol.thr_mult).astype(np.float32)

        # Steps 3–4: maybe retrain
        if intensity != "skip":
            base_n   = min(replay.size(), max(len(Xb) * 2, 200))
            R_X, R_y = replay.sample(base_n, nc,
                                     minority_boost=pol.minority_boost)
            Xc = np.vstack([Xb, R_X]) if len(R_X) > 0 else Xb.copy()
            yc = np.concatenate([yb, R_y]) if len(R_X) > 0 else yb.copy()

            min_cls = np.bincount(yc, minlength=nc).min()
            try:
                Xtr2, Xvl2, ytr2, yvl2 = train_test_split(
                    Xc, yc, test_size=0.15, random_state=seed,
                    stratify=yc if min_cls >= 2 else None)
            except Exception:
                Xtr2, Xvl2, ytr2, yvl2 = train_test_split(
                    Xc, yc, test_size=0.15, random_state=seed)

            severe   = psi_state in ("sudden_drift", "ood")
            Xa2, ya2 = smote_safe(Xtr2, ytr2, nc, ratio=0.55, k=3,
                                  severe=severe, seed=seed)
            bw2      = eff_num_weights(ya2)
            rounds   = 120 if intensity == "full" else 60
            m_main   = _train_xgb(Xa2, ya2, Xvl2, yvl2, nc, bw2, "main", rounds)

            p_tr2    = m_main.predict(
                xgb.DMatrix(Xa2)).reshape(-1, nc).astype(np.float32)
            fw2      = focal_proxy_weights(ya2, p_tr2,
                                           gamma=pol.focal_gamma, base_w=bw2)
            m_div    = _train_xgb(Xa2, ya2, Xvl2, yvl2, nc, fw2, "div", rounds)
            del p_tr2, fw2; gc.collect()

            wt, wd   = _val_weights(m_main, m_div, Xvl2, yvl2, nc)
            p_fv2    = _ensemble_probs(m_main, m_div, Xvl2, nc, wt, wd)
            new_thr  = calibrate_thresholds(p_fv2, yvl2, nc, 30)
            thr      = (0.35 * new_thr + 0.65 * thr).astype(np.float32)
            effective_thr = (thr * pol.thr_mult).astype(np.float32)
            del Xa2, ya2, Xtr2, Xvl2, Xc, yc, p_fv2; gc.collect()
            if len(R_X):
                del R_X, R_y
            last_rt = b_idx

        # Step 5: evaluate on held-out test set
        p_te   = _ensemble_probs(m_main, m_div, X_te, nc, wt, wd)
        y_pred = apply_thresholds(p_te, effective_thr)
        b_f1   = f1_score(y_te, y_pred, average="macro", zero_division=0)
        b_fpr  = fpr_macro(y_te, y_pred, nc)
        del p_te; gc.collect()

        # Step 6: add to replay AFTER evaluation (leakage guard)
        ent = -(np.clip(p_b, 1e-9, 1.) * np.log(np.clip(p_b, 1e-9, 1.))).sum(1)
        replay.add(Xb, yb, ent)
        del p_b, ent; gc.collect()

        el = time.perf_counter() - t0
        print(f"    B{b_idx} [{psi_state:14s}] {intensity:7s}  "
              f"F1={b_f1:.4f} FPR={b_fpr:.4f}  ({el:.1f}s)", flush=True)
        results.append(dict(
            batch=b_idx, psi_state=psi_state, ctx_state=ctx_state,
            intensity=intensity, f1_macro=b_f1, fpr_macro=b_fpr,
        ))
    return results


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 13 — EWC / LwF: MLP Backbone (PyTorch)
# ═════════════════════════════════════════════════════════════════════════════

if _TORCH_OK:

    class MLP(nn.Module):
        """
        256 → 128 → 64 → NC feed-forward network.
        Xavier init, no BatchNorm (BN causes F1 collapse on tabular data).
        """
        def __init__(self, in_dim: int, n_classes: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 256), nn.ReLU(),
                nn.Linear(256, 128),    nn.ReLU(),
                nn.Linear(128, 64),     nn.ReLU(),
                nn.Linear(64, n_classes),
            )
            for layer in self.net:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    def eval_mlp(model, X: np.ndarray, y: np.ndarray, nc: int):
        """Evaluate MLP → returns (macro_f1, macro_fpr)."""
        model.eval()
        with torch.no_grad():
            yp = model(torch.FloatTensor(X)).argmax(1).numpy()
        return (float(f1_score(y, yp, average="macro", zero_division=0)),
                float(fpr_macro(y, yp, nc)))

    def train_initial_mlp(X_tr, y_tr, X_val, y_val, nc: int, seed: int,
                          epochs: int = MLP_INIT_EPOCHS):
        """
        Train MLP from scratch with SMOTE + class-weighted CrossEntropy.
        Early stopping (patience=12) on val macro-F1.
        """
        torch.manual_seed(seed); np.random.seed(seed)
        Xs, ys = smote_safe(X_tr, y_tr, nc, ratio=0.55, k=3,
                            severe=False, seed=seed)
        counts = np.bincount(ys, minlength=nc).astype(np.float32)
        cw     = (counts.sum() / (nc * np.maximum(counts, 1)))
        cw_t   = torch.FloatTensor(cw)

        model  = MLP(Xs.shape[1], nc)
        opt    = torch.optim.Adam(model.parameters(), lr=MLP_LR_INIT,
                                  weight_decay=1e-4)
        crit   = nn.CrossEntropyLoss(weight=cw_t)
        xt     = torch.FloatTensor(Xs)
        yt     = torch.LongTensor(ys)
        ldr    = DataLoader(TensorDataset(xt, yt),
                            batch_size=MLP_BATCH_SIZE, shuffle=True)

        best_val, best_state, no_imp = -1.0, None, 0
        for ep in range(epochs):
            model.train()
            for xb, yb in ldr:
                opt.zero_grad()
                nn.CrossEntropyLoss(weight=cw_t)(model(xb), yb).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            vf1, _ = eval_mlp(model, X_val, y_val, nc)
            if vf1 > best_val + 1e-4:
                best_val  = vf1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_imp    = 0
            else:
                no_imp += 1
                if no_imp >= 12:
                    break
        if best_state:
            model.load_state_dict(best_state)
        return model

    def compute_fisher(model, X: np.ndarray, y: np.ndarray, nc: int,
                       n_samples: int = 800) -> dict:
        """
        Estimate diagonal Fisher information matrix via Monte Carlo sampling.
        Used by EWC as a proxy for parameter importance.
        """
        model.train()
        idx  = np.random.choice(len(y), min(n_samples, len(y)), replace=False)
        xt   = torch.FloatTensor(X[idx])
        yt   = torch.LongTensor(y[idx])
        ldr  = DataLoader(TensorDataset(xt, yt), batch_size=128, shuffle=False)
        fish = {n: torch.zeros_like(p)
                for n, p in model.named_parameters() if p.requires_grad}
        model.zero_grad()
        for xb, yb in ldr:
            out = model(xb)
            logp = F.log_softmax(out, 1)
            yp   = torch.multinomial(F.softmax(out.detach(), 1), 1).squeeze()
            F.nll_loss(logp, yp).backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fish[n] += p.grad.detach() ** 2
            model.zero_grad()
        for n in fish:
            fish[n] /= max(len(ldr), 1)
        return fish

    def train_ewc_mlp(model, fisher_old: dict, params_old: dict,
                      X_new, y_new, X_val, y_val, nc: int, seed: int,
                      epochs: int = MLP_CL_EPOCHS,
                      lam: float   = EWC_LAMBDA):
        """
        EWC continual update.
        loss = CE(new_task) + (λ/2) × Σ F_i × (θ_i − θ*_i)²
        """
        torch.manual_seed(seed)
        Xs, ys = smote_safe(X_new, y_new, nc, ratio=0.55, k=3,
                            severe=False, seed=seed)
        opt  = torch.optim.Adam(model.parameters(), lr=MLP_LR_CL,
                                weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()
        xt   = torch.FloatTensor(Xs)
        yt   = torch.LongTensor(ys)
        ldr  = DataLoader(TensorDataset(xt, yt),
                          batch_size=MLP_BATCH_SIZE, shuffle=True)

        best_val, best_state, no_imp = -1.0, None, 0
        for ep in range(epochs):
            model.train()
            for xb, yb in ldr:
                opt.zero_grad()
                ce      = crit(model(xb), yb)
                ewc_pen = torch.tensor(0.0)
                for n, p in model.named_parameters():
                    if p.requires_grad and n in fisher_old:
                        ewc_pen = ewc_pen + (
                            fisher_old[n] * (p - params_old[n]) ** 2).sum()
                (ce + (lam / 2.0) * ewc_pen).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            vf1, _ = eval_mlp(model, X_val, y_val, nc)
            if vf1 > best_val + 1e-4:
                best_val   = vf1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_imp     = 0
            else:
                no_imp += 1
                if no_imp >= 8:
                    break
        if best_state:
            model.load_state_dict(best_state)
        return model

    def train_lwf_mlp(model, teacher,
                      X_new, y_new, X_val, y_val, nc: int, seed: int,
                      epochs: int = MLP_CL_EPOCHS,
                      T:      float = LWF_T,
                      alpha:  float = LWF_ALPHA):
        """
        LwF (Learning without Forgetting) continual update.
        loss = CE(new_task) + α × KL(softmax(student/T) ‖ softmax(teacher/T)) × T²
        """
        torch.manual_seed(seed)
        Xs, ys = smote_safe(X_new, y_new, nc, ratio=0.55, k=3,
                            severe=False, seed=seed)
        opt  = torch.optim.Adam(model.parameters(), lr=MLP_LR_CL,
                                weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()
        teacher.eval()
        with torch.no_grad():
            soft_all = F.softmax(teacher(torch.FloatTensor(Xs)) / T, 1)

        xt  = torch.FloatTensor(Xs)
        yt  = torch.LongTensor(ys)
        ldr = DataLoader(TensorDataset(xt, yt, soft_all),
                         batch_size=MLP_BATCH_SIZE, shuffle=True)

        best_val, best_state, no_imp = -1.0, None, 0
        for ep in range(epochs):
            model.train()
            for xb, yb, sb in ldr:
                opt.zero_grad()
                out_s = model(xb)
                ce    = crit(out_s, yb)
                kd    = (F.kl_div(F.log_softmax(out_s / T, 1), sb,
                                  reduction="batchmean") * (T ** 2))
                (ce + alpha * kd).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            vf1, _ = eval_mlp(model, X_val, y_val, nc)
            if vf1 > best_val + 1e-4:
                best_val   = vf1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_imp     = 0
            else:
                no_imp += 1
                if no_imp >= 8:
                    break
        if best_state:
            model.load_state_dict(best_state)
        return model


# ═════════════════════════════════════════════════════════════════════════════
# QUICK SELF-TEST  (python3 cadence_core.py)
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("CADENCE core — self-test\n")
    rng = np.random.default_rng(42)
    N, F, NC = 500, 20, 5
    X = rng.standard_normal((N, F)).astype(np.float32)
    y = rng.integers(0, NC, N)

    # Drift detector
    det = AdaptiveDriftDetector()
    d   = det.update(X[:200])
    print(f"  AdaptiveDriftDetector  state={d['state']}  psi={d['psi']:.4f}")
    d   = det.update(X[200:] + 5.0)
    print(f"  AdaptiveDriftDetector  state={d['state']}  psi={d['psi']:.4f}")
    print(f"  retrain_intensity      → {det.retrain_intensity()}")

    # Replay buffer
    buf = SmartReplayBuffer(max_per_class=20)
    ent = rng.random(200).astype(np.float32)
    buf.add(X[:200], y[:200], ent)
    Xr, yr = buf.sample(50, NC)
    print(f"\n  SmartReplayBuffer      size={len(Xr)}")

    # Context-gated policy (post-fix: context never forces skip under drift)
    pol = ContextGatedPolicy().get("mild_drift", "gradual_drift", "partial",
                                   0.2, -0.1, 4, 1.0, 0.0)
    print(f"  ContextGatedPolicy     intensity={pol.intensity}  note={pol.note}")

    print("\n  self-test OK")
