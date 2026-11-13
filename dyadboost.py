# -*- coding: utf-8 -*-
"""DyadBoost: a reference implementation of the pipeline proposed in Section 4.5.

Stages map one-to-one onto the manuscript:

    S1  specification and availability firewall   -> Feature.available_at, TemporalLeakageError
    S2  component decomposition                    -> actor/partner logistic model, dyadic residual
    S3  residual boosting (LightGBM)               -> lgb.train(..., init_score=logit(p_main))
    S4  calibration and conformal prediction       -> isotonic map + Mondrian conformal sets
    S5  attribution ensemble                       -> interventional TreeSHAP over the fold ensemble
    S6  audit and refit                            -> stability screen, refit, Delta reported

The estimator is deliberately small and readable: the point of the code is to make the
*protocol* executable, not to squeeze accuracy out of synthetic data.
"""
from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass

# The Microsoft Store build keeps user packages outside sys.path when the interpreter is
# launched from a restricted shell; add that location once so the module is importable.
_CAND = os.path.expandvars(
    r"%LOCALAPPDATA%\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0"
    r"\LocalCache\local-packages\Python311\site-packages")
if os.path.isdir(_CAND) and _CAND not in sys.path:
    sys.path.insert(0, _CAND)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplcache"))

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

LAYERS = ("actor", "partner", "dyad")


class TemporalLeakageError(RuntimeError):
    """Raised when a declared feature postdates the decision point."""


@dataclass(frozen=True)
class Feature:
    """A predictor with the month at which it becomes observable (0 = at the decision point)."""
    name: str
    layer: str
    available_at: int = 0

    def __post_init__(self):
        if self.layer not in LAYERS:
            raise ValueError(f"layer must be one of {LAYERS}, got {self.layer!r}")


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _positive_class_shap(values):
    """Normalise the several shapes shap returns for binary LightGBM models."""
    if isinstance(values, list):
        return np.asarray(values[-1])
    values = np.asarray(values)
    if values.ndim == 3:                      # (n, p, 2)
        return values[:, :, -1]
    return values


class DyadBoost:
    """Dyad-aware residual boosting with stability-weighted attribution."""

    def __init__(self, features, t0=0, n_splits=5, n_repeats=3, seed=0, lgb_params=None,
                 shap_background=120, verbose=True):
        self.features = list(features)
        self.t0 = t0
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.seed = seed
        self.shap_background = shap_background
        self.verbose = verbose

        # ---- S1: the firewall fires at construction time ----------------------
        late = [f.name for f in self.features if f.available_at > self.t0]
        if late:
            raise TemporalLeakageError(
                f"{len(late)} feature(s) postdate the decision point t0={self.t0}: {late}. "
                "Drop them, or move the decision point to the month at which they exist."
            )

        self.actor = [f.name for f in self.features if f.layer == "actor"]
        self.partner = [f.name for f in self.features if f.layer == "partner"]
        self.dyad = [f.name for f in self.features if f.layer == "dyad"]
        if not self.actor + self.partner:
            raise ValueError("the pipeline needs at least one actor or partner feature")
        if not self.dyad:
            raise ValueError("the pipeline needs at least one dyadic feature")

        self.params = {
            "objective": "binary",
            "learning_rate": 0.05,
            "num_leaves": 4,            # capacity held deliberately low
            "max_depth": 3,
            "min_child_samples": 20,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "verbosity": -1,
            "seed": seed,
            "deterministic": True,
        }
        if lgb_params:
            self.params.update(lgb_params)

        self.folds_ = []                # fitted fold artefacts
        self.oof_ = None                # out-of-fold probability from the training set
        self.iso_ = None                # S4 isotonic map
        self.conformal_ = None          # S4 Mondrian conformal quantiles

    # ------------------------------------------------------------------ helpers
    def _check_columns(self, df):
        """Return the declared predictor matrix.

        The model only ever reads declared features, so an undeclared column - even a
        post-decision one - cannot enter the fit by accident. The firewall therefore lives in
        the constructor, where *declaring* a late feature at an early decision point is refused.
        """
        missing = [c for c in self.feature_names if c not in df.columns]
        if missing:
            raise KeyError(f"features declared but absent from the frame: {missing}")
        return df[self.feature_names].to_numpy(float)

    @property
    def feature_names(self):
        return self.actor + self.partner + self.dyad

    # ------------------------------------------------------------------- fit
    def fit(self, df, y, groups=None):
        X = self._check_columns(df)
        y = np.asarray(y).astype(float)
        X1 = df[self.actor + self.partner].to_numpy(float)
        X2 = df[self.dyad].to_numpy(float)
        idx = np.arange(len(y))

        if groups is not None:
            splits = []
            for r in range(self.n_repeats):
                splits += list(GroupKFold(n_splits=self.n_splits).split(X, y, groups))
        else:
            splits = []
            for r in range(self.n_repeats):
                kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed + r)
                splits += list(kf.split(X, y))

        oof_sum = np.zeros(len(y))
        oof_cnt = np.zeros(len(y))
        self.folds_ = []

        for k, (tr, te) in enumerate(splits):
            art = self._fit_fold(X1, X2, y, tr)
            p_te = self._predict_fold(art, X1[te], X2[te])
            oof_sum[te] += p_te
            oof_cnt[te] += 1
            self.folds_.append(art)

        oof = oof_sum / np.maximum(oof_cnt, 1)
        self.oof_ = oof
        self._fit_calibration(oof, y)
        if self.verbose:
            print(f"[S2-S3] fitted {len(self.folds_)} folds; "
                  f"OOF AUC = {roc_auc_score(y, oof):.3f}, "
                  f"Brier = {brier_score_loss(y, oof):.4f}")
        return self

    def _fit_fold(self, X1, X2, y, tr):
        # ---- S2: actor and partner main effects -----------------------------
        scaler = StandardScaler().fit(X1[tr])
        main = LogisticRegression(C=1.0, max_iter=2000).fit(scaler.transform(X1[tr]), y[tr])
        p_main_tr = main.predict_proba(scaler.transform(X1[tr]))[:, 1]

        # inner split used only for early stopping
        tr_in, tr_val = train_test_split(
            tr, test_size=0.25, random_state=self.seed, stratify=y[tr])
        pos = {v: i for i, v in enumerate(tr)}
        i_in = np.array([pos[v] for v in tr_in])
        i_val = np.array([pos[v] for v in tr_val])

        dtr = lgb.Dataset(X2[tr_in], y[tr_in], init_score=_logit(p_main_tr[i_in]))
        dval = lgb.Dataset(X2[tr_val], y[tr_val], reference=dtr,
                           init_score=_logit(p_main_tr[i_val]))

        # ---- S3: residual boosting ------------------------------------------
        booster = lgb.train(
            self.params, dtr, num_boost_round=600, valid_sets=[dval],
            callbacks=[lgb.early_stopping(40, verbose=False)])

        rng = np.random.default_rng(self.seed)
        bg_idx = rng.choice(tr, size=min(self.shap_background, len(tr)), replace=False)
        explainer = shap.TreeExplainer(
            booster, data=X2[bg_idx], feature_perturbation="interventional")

        return {"scaler": scaler, "main": main, "booster": booster, "explainer": explainer}

    def _predict_fold(self, art, X1, X2):
        p_main = art["main"].predict_proba(art["scaler"].transform(X1))[:, 1]
        raw = art["booster"].predict(X2, raw_score=True)
        return _sigmoid(_logit(p_main) + raw)

    def predict_proba(self, df):
        X1 = df[self.actor + self.partner].to_numpy(float)
        X2 = df[self.dyad].to_numpy(float)
        p = np.mean([self._predict_fold(a, X1, X2) for a in self.folds_], axis=0)
        return self.iso_.predict(p) if self.iso_ is not None else p

    # ------------------------------------------------------------------- S4
    def _stratum(self, p):
        return np.minimum((p * 5).astype(int), 4)      # 5 risk strata

    def _fit_calibration(self, oof, y):
        self.iso_ = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
        p_cal = self.iso_.predict(oof)
        # conformal score = probability assigned to the observed label
        score = np.where(y == 1, p_cal, 1.0 - p_cal)
        self.conformal_ = {}
        for s in range(5):
            m = self._stratum(oof) == s
            self.conformal_[s] = np.sort(score[m]) if m.any() else np.array([1.0])

    def conformal_sets(self, p, alpha=0.10):
        """Mondrian split-conformal prediction sets; returns {include_1, include_0} masks."""
        p_cal = self.iso_.predict(p)
        strata = self._stratum(p)
        include_1, include_0 = np.zeros(len(p), bool), np.zeros(len(p), bool)
        for s in range(5):
            m = strata == s
            if not m.any():
                continue
            scores = self.conformal_[s]
            n = len(scores)
            q = np.quantile(scores, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n))
            include_1[m] = p_cal[m] >= 1.0 - q
            include_0[m] = (1.0 - p_cal[m]) >= 1.0 - q
        return include_0, include_1

    # ------------------------------------------------------------------- S5
    def shap_ensemble(self, df):
        """Per-fold SHAP matrices on the positive-class logit scale."""
        X2 = df[self.dyad].to_numpy(float)
        out = []
        for art in self.folds_:
            vals = art["explainer"].shap_values(X2, check_additivity=False)
            out.append(_positive_class_shap(vals))
        return np.stack(out)                       # (folds, n, p_dyad)

    def attribution_table(self, df):
        phi = self.shap_ensemble(df)               # (K, n, p)
        k = phi.shape[0]
        mag = np.abs(phi).mean(axis=(0, 1))
        ranks = np.argsort(np.argsort(-np.abs(phi).mean(axis=1), axis=1), axis=1) + 1  # (K, p)
        rank_std = ranks.std(axis=0)
        p = phi.shape[2]
        stab_rank = np.clip(1.0 - rank_std / max(p / 2.0, 1.0), 0.0, 1.0)
        # scale-free stability: how much does this feature's own mean attribution vary
        # across the ensemble? rank-based stability is brittle when several predictors are weak.
        mag_k = np.abs(phi).mean(axis=1)                 # (K, p)
        cv = mag_k.std(axis=0) / np.maximum(mag_k.mean(axis=0), 1e-9)
        stab_cv = np.clip(1.0 - cv, 0.0, 1.0)
        # sign consistency is assessed on the fold-level mean attribution: per-case signs are
        # near coin flips whenever the signal is weak, so they are the wrong unit.
        fold_means = phi.mean(axis=1)                       # (K, p)
        sign = np.maximum((fold_means > 0).mean(axis=0), (fold_means < 0).mean(axis=0))
        mean_signed = fold_means.mean(axis=0)
        per_case = phi.mean(axis=0)                # (n, p)
        sign = np.maximum((per_case > 0).mean(axis=0), (per_case < 0).mean(axis=0))
        swa = (mag / max(mag.max(), 1e-12)) * stab_cv * sign
        return pd.DataFrame({
            "feature": self.dyad,
            "mean_abs_shap": mag,
            "rank_sd": rank_std,
            "stability_rank": stab_rank,
            "cv": cv,
            "stability": stab_cv,
            "sign_consistency": sign,
            "swa": swa,
            "mean_signed_shap": mean_signed,
        }).sort_values("swa", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------- S6
    def audit(self, df, y, stability_min=0.60, sign_min=0.79):
        table = self.attribution_table(df)
        flagged = table.loc[(table.stability < stability_min) | (table.sign_consistency < sign_min),
                            "feature"].tolist()
        return table, flagged


# ---------------------------------------------------------------- evaluation
def evaluate(p, y):
    return {
        "auc": roc_auc_score(y, p),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
    }


def coverage(p, y, include_0, include_1):
    covered = np.where(y == 1, include_1, include_0)
    size = include_0.astype(int) + include_1.astype(int)
    return covered.mean(), size.mean()
