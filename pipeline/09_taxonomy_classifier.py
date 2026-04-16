#!/usr/bin/env python3
"""GASP Phase 09 v2: Calibrated RF taxonomy classifier with Optuna-tuned parameters.

Trains a CalibratedRandomForestClassifier on 358 Mahlke et al. (2022) labeled
asteroids. Feature Set: 16 Gaia reflectance bands + spectral slopes s1–s4 (20
features, 100 % coverage on refl_418–990).

Hyperparameters are loaded from data/final/optuna_best_params.json by default
(pre-tuned: 50-trial Optuna study, F1-macro objective, StratifiedKFold k=5).
Use --no-skip-optuna to re-run the full search (~2.5 h on VPS).

A model comparison (RF / XGB / LGBM / Stacking) is always reported via 5-fold
cross_val_predict so quality can be monitored on each run.

X-complex post-hoc refinement (Tholen 1984):
  Predictions of "X" with NEOWISE albedo available are replaced by:
    E : albedo > 0.30   (enstatite-like, high albedo)
    M : 0.10 ≤ albedo ≤ 0.30  (metallic)
    P : albedo < 0.10   (primitive, low albedo)

Output columns added / overwritten in catalog:
  taxonomy_ml           — predicted class (C/S/X/V/D or E/M/P)
  taxonomy_ml_prob      — max class probability (calibrated RF)
  taxonomy_ml_source    — "gasp_rf_v2" for all predictions
  taxonomy_ml_x_refined — True if X→E/M/P albedo rule was applied
  taxonomy_final        — Mahlke label if available, else ML (prob ≥ 0.70)

Usage:
  python pipeline/09_taxonomy_classifier.py                 # default: load saved params
  python pipeline/09_taxonomy_classifier.py --no-skip-optuna  # re-run Optuna (slow)
"""

from __future__ import annotations

import os
import sys

# Ensure line-buffered output even when stdout is piped (e.g. | tee).
os.environ["PYTHONUNBUFFERED"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import argparse
import json
import shutil
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
FINAL_DIR   = Path("data/final")
FIGURES_DIR = Path("figures")

CATALOG_FILE = FINAL_DIR / "gasp_catalog_v1.parquet"
LOG_FILE     = FINAL_DIR / "09_taxonomy_classifier_log.json"
PARAMS_FILE  = FINAL_DIR / "optuna_best_params.json"
CM_FIGURE    = FIGURES_DIR / "09_cm_model_a.png"

# ── features ───────────────────────────────────────────────────────────────────
BANDS        = [374, 418, 462, 506, 550, 594, 638, 682,
                726, 770, 814, 858, 902, 946, 990, 1034]
REFL_COLS    = [f"refl_{b}" for b in BANDS]
SLOPE_COLS   = ["s1", "s2", "s3", "s4"]
FEATURE_COLS = REFL_COLS + SLOPE_COLS   # 20 features

# ── X-complex albedo thresholds for post-hoc subclassification (Tholen 1984) ──
X_ALBEDO_E_MIN = 0.30   # E-type: albedo > 0.30
X_ALBEDO_P_MAX = 0.10   # P-type: albedo < 0.10  (M-type: 0.10–0.30)

# ── model / search parameters ──────────────────────────────────────────────────
CONF_THRESHOLD  = 0.70
CV_FOLDS        = 5
OPTUNA_N_TRIALS = 50
RANDOM_STATE    = 42
SOURCE_TAG      = "gasp_rf_v2"

# ── taxonomy mapping (Mahlke et al. 2022 → broad complex) ─────────────────────
COMPLEX_MAP: dict[str, str] = {
    "S":  "S", "A":  "S", "K":  "S", "L":  "S", "Q":  "S",
    "C":  "C", "B":  "C", "Ch": "C",
    "X":  "X", "Xe": "X", "Xk": "X", "M":  "X", "Mk": "X", "P":  "X", "Pk": "X",
    "V":  "V",
    "D":  "D", "Z":  "D",
}


# ── XGBoost wrapper: auto-injects balanced sample weights ─────────────────────
class BalancedXGBClassifier(XGBClassifier):
    def fit(self, X, y, **kwargs):
        kwargs.setdefault("sample_weight", compute_sample_weight("balanced", y))
        kwargs.setdefault("verbose", False)
        return super().fit(X, y, **kwargs)


# ── helpers ────────────────────────────────────────────────────────────────────

def _impute_median(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())
    return df


def _x_subclass(albedo: float) -> str:
    """Map X-complex albedo to E / M / P subtype (Tholen 1984)."""
    if albedo >= X_ALBEDO_E_MIN:
        return "E"
    if albedo < X_ALBEDO_P_MAX:
        return "P"
    return "M"


def _make_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


# ── model builders ─────────────────────────────────────────────────────────────

def _build_rf(params: dict | None = None) -> RandomForestClassifier:
    defaults = dict(n_estimators=500, max_features="sqrt", min_samples_leaf=2,
                    class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    return RandomForestClassifier(**(defaults | (params or {})))


def _build_xgb(params: dict | None = None) -> BalancedXGBClassifier:
    defaults = dict(n_estimators=300, max_depth=5, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                    random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
    return BalancedXGBClassifier(**(defaults | (params or {})))


def _build_lgbm(params: dict | None = None) -> LGBMClassifier:
    defaults = dict(n_estimators=300, max_depth=7, learning_rate=0.1,
                    num_leaves=31, min_child_samples=10, subsample=0.8,
                    colsample_bytree=0.8, class_weight="balanced",
                    random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    return LGBMClassifier(**(defaults | (params or {})))


# ── Optuna objective factories ─────────────────────────────────────────────────

def _rf_objective(X: np.ndarray, y: np.ndarray, cv: StratifiedKFold):
    def objective(trial: optuna.Trial) -> float:
        p = dict(
            n_estimators     = trial.suggest_int("n_estimators", 100, 1000),
            max_depth        = trial.suggest_int("max_depth", 3, 20),
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10),
            max_features     = trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        )
        m = RandomForestClassifier(**p)
        return float(np.mean([
            f1_score(y[v], m.fit(X[t], y[t]).predict(X[v]), average="macro")
            for t, v in cv.split(X, y)
        ]))
    return objective


def _xgb_objective(X: np.ndarray, y: np.ndarray, cv: StratifiedKFold):
    def objective(trial: optuna.Trial) -> float:
        p = dict(
            n_estimators     = trial.suggest_int("n_estimators", 100, 500),
            max_depth        = trial.suggest_int("max_depth", 3, 10),
            learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample        = trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight = trial.suggest_int("min_child_weight", 1, 10),
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
        )
        m = BalancedXGBClassifier(**p)
        return float(np.mean([
            f1_score(y[v], m.fit(X[t], y[t]).predict(X[v]), average="macro")
            for t, v in cv.split(X, y)
        ]))
    return objective


def _lgbm_objective(X: np.ndarray, y: np.ndarray, cv: StratifiedKFold):
    def objective(trial: optuna.Trial) -> float:
        p = dict(
            n_estimators      = trial.suggest_int("n_estimators", 100, 500),
            max_depth         = trial.suggest_int("max_depth", 3, 15),
            learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            num_leaves        = trial.suggest_int("num_leaves", 15, 127),
            min_child_samples = trial.suggest_int("min_child_samples", 5, 50),
            subsample         = trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.6, 1.0),
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        )
        m = LGBMClassifier(**p)
        return float(np.mean([
            f1_score(y[v], m.fit(X[t], y[t]).predict(X[v]), average="macro")
            for t, v in cv.split(X, y)
        ]))
    return objective


def _run_optuna(name: str, objective, n_trials: int) -> tuple[dict, float]:
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  {name}: F1-macro={study.best_value:.4f}  params={study.best_params}")
    return study.best_params, study.best_value


# ── CV evaluation ──────────────────────────────────────────────────────────────

def _eval_cv(
    model,
    X: np.ndarray,
    y_enc: np.ndarray,
    le: LabelEncoder,
    label: str,
    cm_path: Path | None,
    plt_mod,
) -> dict:
    """cross_val_predict → metrics dict + optional confusion matrix figure."""
    classes   = le.classes_
    y_pred    = cross_val_predict(model, X, y_enc, cv=_make_cv())
    y_labels  = le.inverse_transform(y_enc)
    yp_labels = le.inverse_transform(y_pred)

    acc    = float((y_pred == y_enc).mean())
    f1_mac = float(f1_score(y_enc, y_pred, average="macro"))
    per_class = {
        cls: round(float(f1_score(y_enc, y_pred,
                                   labels=[int(np.where(classes == cls)[0][0])],
                                   average="macro")), 4)
        for cls in classes
    }

    print(f"  Accuracy : {acc*100:.1f}%")
    print(f"  F1-macro : {f1_mac:.4f}")
    print(f"  F1/class : " + "  ".join(f"{c}={per_class[c]:.3f}" for c in classes))
    print()
    rep = classification_report(y_labels, yp_labels, target_names=classes, zero_division=0)
    for line in rep.splitlines():
        print(f"    {line}")
    print()

    if cm_path is not None:
        fig, ax = plt_mod.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay.from_predictions(
            y_labels, yp_labels, display_labels=classes,
            cmap="Blues", colorbar=False, ax=ax,
        )
        ax.set_title(
            f"GASP — {label} — {CV_FOLDS}-fold CV\n"
            f"Acc={acc*100:.1f}%  F1-macro={f1_mac:.4f}  (n={len(y_enc)})",
            fontsize=11,
        )
        plt_mod.tight_layout()
        fig.savefig(cm_path, dpi=150, bbox_inches="tight")
        plt_mod.close(fig)
        print(f"  CM → {cm_path}")
        print()

    return {"label": label, "n": int(len(y_enc)),
            "accuracy": round(acc, 4), "f1_macro": round(f1_mac, 4),
            "f1_class": per_class}


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    t0 = time.time()

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--skip-optuna", default=True,
        action=argparse.BooleanOptionalAction,
        help="Load saved Optuna params instead of re-running search (default: True)",
    )
    args = parser.parse_args()

    if not CATALOG_FILE.is_file():
        print(f"ERROR: catalog not found: {CATALOG_FILE}", file=sys.stderr)
        return 1

    FIGURES_DIR.mkdir(exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=== GASP Phase 09 v2: Taxonomy Classifier ===")
    print(f"Input        : {CATALOG_FILE}")
    print(f"Threshold    : {CONF_THRESHOLD}")
    print(f"Skip Optuna  : {args.skip_optuna}"
          + ("  (loading saved params)" if args.skip_optuna else
             f"  (running {OPTUNA_N_TRIALS} trials × 3 models)"))
    print()

    # ── load catalog ───────────────────────────────────────────────────────────
    df = pd.read_parquet(CATALOG_FILE)
    df = df.drop(columns=[c for c in ["complex", "taxonomy_ml", "taxonomy_ml_prob",
                                       "taxonomy_ml_source", "taxonomy_ml_x_refined",
                                       "taxonomy_final"] if c in df.columns])
    n_total = len(df)
    print(f"Loaded : {n_total:,} asteroids, {df.shape[1]} columns")

    df["complex"] = df["taxonomy"].map(COMPLEX_MAP)
    lab_mask  = df["complex"].notna()
    n_labeled = int(lab_mask.sum())
    print(f"Labeled: {n_labeled}")
    for cls, cnt in df.loc[lab_mask, "complex"].value_counts().items():
        print(f"  {cls}: {cnt:3d}  ({cnt/n_labeled*100:.1f}%)")
    print()

    # ── feature check + imputation ─────────────────────────────────────────────
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"ERROR: missing feature columns: {missing}", file=sys.stderr)
        return 1
    df_imp = _impute_median(df.copy(), FEATURE_COLS)

    X      = df_imp.loc[lab_mask, FEATURE_COLS].values.astype(np.float32)
    y      = df.loc[lab_mask, "complex"].values
    le     = LabelEncoder().fit(y)
    y_enc  = le.transform(y)
    cv     = _make_cv()

    # ── Optuna / load params ───────────────────────────────────────────────────
    print("─" * 60)
    if args.skip_optuna:
        if not PARAMS_FILE.is_file():
            print(f"ERROR: --skip-optuna requested but {PARAMS_FILE} not found.",
                  file=sys.stderr)
            print("       Run with --no-skip-optuna to generate it.", file=sys.stderr)
            return 1
        saved = json.loads(PARAMS_FILE.read_text())
        rf_params   = saved["rf"]["params"]
        xgb_params  = saved["xgb"]["params"]
        lgbm_params = saved["lgbm"]["params"]
        print(f"Loaded Optuna params from {PARAMS_FILE}  "
              f"(tuned {saved['n_trials']} trials, {saved['timestamp'][:10]})")
        print(f"  RF   best F1-macro: {saved['rf']['best_f1_macro']}")
        print(f"  XGB  best F1-macro: {saved['xgb']['best_f1_macro']}")
        print(f"  LGBM best F1-macro: {saved['lgbm']['best_f1_macro']}")
        optuna_log = saved  # re-use for log
    else:
        print(f"Optuna search — {OPTUNA_N_TRIALS} trials × 3 models (n={n_labeled})")
        t_opt = time.time()
        print("  [1/3] RandomForest …")
        rf_params,   rf_best   = _run_optuna("RF",   _rf_objective(X, y_enc, cv),   OPTUNA_N_TRIALS)
        print("  [2/3] XGBoost …")
        xgb_params,  xgb_best  = _run_optuna("XGB",  _xgb_objective(X, y_enc, cv),  OPTUNA_N_TRIALS)
        print("  [3/3] LightGBM …")
        lgbm_params, lgbm_best = _run_optuna("LGBM", _lgbm_objective(X, y_enc, cv), OPTUNA_N_TRIALS)
        print(f"Optuna done in {time.time()-t_opt:.1f}s")
        optuna_log = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_trials":  OPTUNA_N_TRIALS,
            "rf":   {"best_f1_macro": round(rf_best, 4),   "params": rf_params},
            "xgb":  {"best_f1_macro": round(xgb_best, 4),  "params": xgb_params},
            "lgbm": {"best_f1_macro": round(lgbm_best, 4), "params": lgbm_params},
        }
        PARAMS_FILE.write_text(json.dumps(optuna_log, indent=2), encoding="utf-8")
        print(f"Params saved → {PARAMS_FILE}")
    print()

    # ── CV evaluation — all 4 models ───────────────────────────────────────────
    print("─" * 60)
    print("CV evaluation — 5-fold cross_val_predict")
    print("─" * 60)

    results: dict[str, dict] = {}

    print("\n[RF]")
    results["rf"] = _eval_cv(
        _build_rf({**rf_params, "class_weight": "balanced",
                   "random_state": RANDOM_STATE, "n_jobs": -1}),
        X, y_enc, le, "RandomForest (tuned)", CM_FIGURE, plt,
    )
    print("[XGB]")
    results["xgb"] = _eval_cv(
        _build_xgb({**xgb_params, "random_state": RANDOM_STATE,
                    "n_jobs": -1, "verbosity": 0}),
        X, y_enc, le, "XGBoost (tuned)", None, plt,
    )
    print("[LGBM]")
    results["lgbm"] = _eval_cv(
        _build_lgbm({**lgbm_params, "class_weight": "balanced",
                     "random_state": RANDOM_STATE, "n_jobs": -1, "verbose": -1}),
        X, y_enc, le, "LightGBM (tuned)", None, plt,
    )
    print("[Stacking: RF + XGB + LGBM → LogisticRegression]")
    results["stack"] = _eval_cv(
        StackingClassifier(
            estimators=[
                ("rf",   _build_rf({**rf_params, "class_weight": "balanced",
                                    "random_state": RANDOM_STATE, "n_jobs": -1})),
                ("xgb",  _build_xgb({**xgb_params, "random_state": RANDOM_STATE,
                                      "n_jobs": -1, "verbosity": 0})),
                ("lgbm", _build_lgbm({**lgbm_params, "class_weight": "balanced",
                                       "random_state": RANDOM_STATE, "n_jobs": -1,
                                       "verbose": -1})),
            ],
            final_estimator=LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
            ),
            cv=_make_cv(), n_jobs=-1,
        ),
        X, y_enc, le, "Stacking (RF+XGB+LGBM → LR)", None, plt,
    )

    # comparison table
    print("─" * 60)
    print("Model comparison")
    print("─" * 60)
    classes_list = list(le.classes_)
    hdr = f"{'Model':28s}  {'Acc':>6s}  {'F1-mac':>7s}  " + \
          "  ".join(f"F1-{c}" for c in classes_list)
    print(hdr)
    print("─" * len(hdr))
    for key, r in results.items():
        row = (f"{r['label'][:28]:28s}  {r['accuracy']*100:5.1f}%  "
               f"{r['f1_macro']:.4f}   " +
               "  ".join(f"{r['f1_class'].get(c, 0):.3f}" for c in classes_list))
        print(row)
    print()

    best_type = max(results, key=lambda k: results[k]["f1_macro"])
    print(f"Winner: {best_type.upper()}  (F1-macro={results[best_type]['f1_macro']:.4f})")
    print()

    # ── train final RF (calibrated) on full labeled set ────────────────────────
    print("─" * 60)
    print(f"Training final CalibratedRF on all {n_labeled} labeled asteroids …")
    final_rf = CalibratedClassifierCV(
        _build_rf({**rf_params, "class_weight": "balanced",
                   "random_state": RANDOM_STATE, "n_jobs": -1}),
        method="isotonic", cv=CV_FOLDS,
    )
    final_rf.fit(X, y_enc)
    print("Done.")
    print()

    # ── predict for all 19,190 ─────────────────────────────────────────────────
    print("Predicting for all asteroids …")
    X_all   = df_imp[FEATURE_COLS].values.astype(np.float32)
    proba   = final_rf.predict_proba(X_all)
    pred    = le.inverse_transform(np.argmax(proba, axis=1))
    prob    = proba.max(axis=1)

    # X→E/M/P post-hoc refinement (rule-based, no separate model)
    alb_vals  = df["albedo"].values
    x_refined = np.zeros(n_total, dtype=bool)
    for i in range(n_total):
        if pred[i] == "X" and np.isfinite(alb_vals[i]) and alb_vals[i] > 0:
            pred[i]      = _x_subclass(float(alb_vals[i]))
            x_refined[i] = True

    df["taxonomy_ml"]           = pred
    df["taxonomy_ml_prob"]      = prob.round(4)
    df["taxonomy_ml_source"]    = SOURCE_TAG
    df["taxonomy_ml_x_refined"] = x_refined

    # taxonomy_final
    mahlke_ok = df["taxonomy"].notna().values
    ml_ok     = prob >= CONF_THRESHOLD
    df["taxonomy_final"] = np.where(
        mahlke_ok, df["taxonomy"].values,
        np.where(ml_ok, pred, np.nan),
    )

    # ── coverage report ────────────────────────────────────────────────────────
    n_mahlke   = int(mahlke_ok.sum())
    n_ml_added = int((~mahlke_ok & ml_ok).sum())
    n_low_conf = int((~mahlke_ok & ~ml_ok).sum())
    n_final    = int(df["taxonomy_final"].notna().sum())
    n_x_ref    = int(x_refined.sum())

    print()
    print("=== Coverage Report ===")
    print(f"  Mahlke labels (original)    : {n_mahlke:,}")
    print(f"  ML added (prob ≥ {CONF_THRESHOLD})       : {n_ml_added:,}")
    print(f"  Low-confidence (excluded)   : {n_low_conf:,}")
    print(f"  taxonomy_final total         : {n_final:,}  "
          f"({n_final/n_total*100:.1f}% of {n_total:,})")
    print(f"  X→E/M/P refined (albedo)    : {n_x_ref:,}")
    print()
    print("taxonomy_final distribution:")
    for cls, cnt in df["taxonomy_final"].value_counts().items():
        print(f"  {cls:4s}: {cnt:6,}  ({cnt/n_total*100:.1f}%)")
    print()

    # ── backup + write ─────────────────────────────────────────────────────────
    bak = CATALOG_FILE.with_suffix(".parquet.bak")
    shutil.copy2(CATALOG_FILE, bak)
    print(f"Backup : {bak}")
    df = df.drop(columns=["complex"])
    df.to_parquet(CATALOG_FILE, index=False, compression="snappy")
    print(f"Saved  : {CATALOG_FILE}  ({df.shape[0]:,} rows × {df.shape[1]} cols)")

    # ── JSON log ───────────────────────────────────────────────────────────────
    log = {
        "timestamp":            datetime.now(timezone.utc).isoformat(),
        "gasp_phase":           "09_taxonomy_classifier_v2",
        "skip_optuna":          args.skip_optuna,
        "winner_model":         best_type,
        "source_tag":           SOURCE_TAG,
        "optuna":               optuna_log,
        "cv_comparison": {
            k: {"f1_macro": r["f1_macro"], "accuracy": r["accuracy"],
                "f1_class": r["f1_class"]}
            for k, r in results.items()
        },
        "confidence_threshold": CONF_THRESHOLD,
        "complex_map":          COMPLEX_MAP,
        "x_albedo_thresholds":  {"E_min": X_ALBEDO_E_MIN, "P_max": X_ALBEDO_P_MAX},
        "coverage": {
            "n_mahlke":            int(n_mahlke),
            "n_ml_added":          int(n_ml_added),
            "n_low_confidence":    int(n_low_conf),
            "n_taxonomy_final":    int(n_final),
            "taxonomy_final_pct":  round(n_final / n_total * 100, 2),
            "n_x_refined":         int(n_x_ref),
        },
        "output_file":    str(CATALOG_FILE),
        "backup_file":    str(bak),
        "params_file":    str(PARAMS_FILE),
        "cm_figure":      str(CM_FIGURE),
        "duration_seconds": round(time.time() - t0, 1),
    }
    LOG_FILE.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"Log    : {LOG_FILE}")
    print()
    print(f"Done in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
