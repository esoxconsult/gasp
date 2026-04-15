#!/usr/bin/env python3
"""GASP Phase 09 v2: Optimized taxonomy classifier — RF + XGB + LGBM + Stacking.

Three candidate models are tuned via Optuna (n_trials=50, F1-macro objective,
StratifiedKFold k=5) and combined in a StackingClassifier. The model with the
highest CV F1-macro is used for final predictions.

Feature routing (same two-model design as v1):
  Model A  20 features  Gaia bands + slopes          all 19,190 asteroids
  Model B  21 features  same + NEOWISE albedo         ~22.3 % (4,286 objects)

X-complex post-hoc albedo subclassification (Tholen 1984):
  E : albedo > 0.30   M : 0.10 ≤ albedo ≤ 0.30   P : albedo < 0.10

Output columns:
  taxonomy_ml        — predicted class (broad complex or E/M/P)
  taxonomy_ml_prob   — max class probability from selected model
  taxonomy_ml_source — "gasp_<model>_v2_A" or "gasp_<model>_v2_B"
  taxonomy_final     — Mahlke label if available, else ML (prob ≥ 0.70)

Usage:
  python pipeline/09_taxonomy_classifier.py
"""

from __future__ import annotations

import json
import shutil
import sys
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
from sklearn.metrics import classification_report, f1_score
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

CATALOG_FILE   = FINAL_DIR / "gasp_catalog_v1.parquet"
LOG_FILE       = FINAL_DIR / "09_taxonomy_classifier_log.json"
PARAMS_FILE    = FINAL_DIR / "optuna_best_params.json"
CM_A_FIGURE    = FIGURES_DIR / "09_cm_model_a.png"
CM_B_FIGURE    = FIGURES_DIR / "09_cm_model_b.png"

# ── feature sets ───────────────────────────────────────────────────────────────
BANDS          = [374, 418, 462, 506, 550, 594, 638, 682,
                  726, 770, 814, 858, 902, 946, 990, 1034]
REFL_COLS      = [f"refl_{b}" for b in BANDS]
SLOPE_COLS     = ["s1", "s2", "s3", "s4"]
FEATURE_COLS_A = REFL_COLS + SLOPE_COLS               # 20 features
FEATURE_COLS_B = REFL_COLS + SLOPE_COLS + ["albedo"]  # 21 features

# ── X-complex albedo thresholds (Tholen 1984) ──────────────────────────────────
X_ALBEDO_E_MIN = 0.30   # E-type (enstatite): albedo > 0.30
X_ALBEDO_P_MAX = 0.10   # P-type (primitive): albedo < 0.10

# ── hyper-parameters ───────────────────────────────────────────────────────────
CONF_THRESHOLD  = 0.70
CV_FOLDS        = 5
OPTUNA_N_TRIALS = 50
RANDOM_STATE    = 42
MIN_B_CLASS_N   = 5       # minimum per-class samples for Model B training

# ── taxonomy mapping (Mahlke et al. 2022 → broad complex) ─────────────────────
COMPLEX_MAP: dict[str, str] = {
    "S":  "S", "A":  "S", "K":  "S", "L":  "S", "Q":  "S",
    "C":  "C", "B":  "C", "Ch": "C",
    "X":  "X", "Xe": "X", "Xk": "X", "M":  "X", "Mk": "X", "P":  "X", "Pk": "X",
    "V":  "V",
    "D":  "D", "Z":  "D",
}


# ── XGBoost wrapper with automatic balanced sample weights ────────────────────
class BalancedXGBClassifier(XGBClassifier):
    """XGBClassifier that auto-injects balanced sample_weight on every fit()."""

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
    if albedo >= X_ALBEDO_E_MIN:
        return "E"
    if albedo < X_ALBEDO_P_MAX:
        return "P"
    return "M"


def _make_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


# ── model builders ─────────────────────────────────────────────────────────────

def _build_rf(params: dict | None = None) -> RandomForestClassifier:
    defaults = dict(
        n_estimators=500, max_features="sqrt", min_samples_leaf=2,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    )
    return RandomForestClassifier(**(defaults | (params or {})))


def _build_xgb(params: dict | None = None) -> BalancedXGBClassifier:
    defaults = dict(
        n_estimators=300, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
    )
    return BalancedXGBClassifier(**(defaults | (params or {})))


def _build_lgbm(params: dict | None = None) -> LGBMClassifier:
    defaults = dict(
        n_estimators=300, max_depth=7, learning_rate=0.1,
        num_leaves=31, min_child_samples=10,
        subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )
    return LGBMClassifier(**(defaults | (params or {})))


# ── Optuna objective factories ─────────────────────────────────────────────────

def _rf_objective(X: np.ndarray, y: np.ndarray, cv: StratifiedKFold):
    def objective(trial: optuna.Trial) -> float:
        p = dict(
            n_estimators     = trial.suggest_int("n_estimators", 100, 1000),
            max_depth        = trial.suggest_int("max_depth", 3, 20),
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10),
            max_features     = trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            class_weight     = "balanced",
            random_state     = RANDOM_STATE,
            n_jobs           = -1,
        )
        m = RandomForestClassifier(**p)
        scores = [
            f1_score(y[v], m.fit(X[t], y[t]).predict(X[v]), average="macro")
            for t, v in cv.split(X, y)
        ]
        return float(np.mean(scores))
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
            random_state     = RANDOM_STATE,
            n_jobs           = -1,
            verbosity        = 0,
        )
        m = BalancedXGBClassifier(**p)
        scores = [
            f1_score(y[v], m.fit(X[t], y[t]).predict(X[v]), average="macro")
            for t, v in cv.split(X, y)
        ]
        return float(np.mean(scores))
    return objective


def _lgbm_objective(X: np.ndarray, y: np.ndarray, cv: StratifiedKFold):
    def objective(trial: optuna.Trial) -> float:
        p = dict(
            n_estimators     = trial.suggest_int("n_estimators", 100, 500),
            max_depth        = trial.suggest_int("max_depth", 3, 15),
            learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            num_leaves       = trial.suggest_int("num_leaves", 15, 127),
            min_child_samples= trial.suggest_int("min_child_samples", 5, 50),
            subsample        = trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0),
            class_weight     = "balanced",
            random_state     = RANDOM_STATE,
            n_jobs           = -1,
            verbose          = -1,
        )
        m = LGBMClassifier(**p)
        scores = [
            f1_score(y[v], m.fit(X[t], y[t]).predict(X[v]), average="macro")
            for t, v in cv.split(X, y)
        ]
        return float(np.mean(scores))
    return objective


def _run_optuna(
    name: str,
    objective,
    n_trials: int,
) -> tuple[dict, float]:
    """Run an Optuna study; return (best_params, best_value)."""
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  {name}: best F1-macro={study.best_value:.4f}  "
          f"params={study.best_params}")
    return study.best_params, study.best_value


# ── model evaluation via cross_val_predict ─────────────────────────────────────

def _eval_cv(
    model,
    X: np.ndarray,
    y_enc: np.ndarray,
    le: LabelEncoder,
    label: str,
    cm_path: Path | None,
    plt_mod,
) -> dict:
    """5-fold CV evaluation → print report + confusion matrix → return metrics."""
    classes   = le.classes_
    cv        = _make_cv()
    y_pred    = cross_val_predict(model, X, y_enc, cv=cv)
    y_labels  = le.inverse_transform(y_enc)
    yp_labels = le.inverse_transform(y_pred)

    acc    = float((y_pred == y_enc).mean())
    f1_mac = float(f1_score(y_enc, y_pred, average="macro"))

    per_class = {}
    for cls in classes:
        idx  = int(np.where(classes == cls)[0][0])
        per_class[cls] = round(
            float(f1_score(y_enc, y_pred, labels=[idx], average="macro")), 4
        )

    rep = classification_report(y_labels, yp_labels,
                                target_names=classes, zero_division=0)
    print(f"  Accuracy : {acc*100:.1f}%")
    print(f"  F1-macro : {f1_mac:.4f}")
    print(f"  F1/class : " +
          "  ".join(f"{c}={per_class[c]:.3f}" for c in classes))
    print()
    for line in rep.splitlines():
        print(f"    {line}")
    print()

    if cm_path is not None:
        fig, ax = plt_mod.subplots(figsize=(7, 6))
        from sklearn.metrics import ConfusionMatrixDisplay
        ConfusionMatrixDisplay.from_predictions(
            y_labels, yp_labels,
            display_labels=classes, cmap="Blues", colorbar=False, ax=ax,
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

    return {
        "label":     label,
        "n":         int(len(y_enc)),
        "accuracy":  round(acc, 4),
        "f1_macro":  round(f1_mac, 4),
        "f1_class":  per_class,
    }


# ── final model factory ────────────────────────────────────────────────────────

def _make_final_model_a(model_type: str, best_params: dict, all_results: dict):
    """Build the calibrated/native final model for Model A (Feature Set A)."""
    if model_type == "stack":
        rf   = _build_rf(best_params.get("rf"))
        xgb  = _build_xgb(best_params.get("xgb"))
        lgbm = _build_lgbm(best_params.get("lgbm"))
        return StackingClassifier(
            estimators=[("rf", rf), ("xgb", xgb), ("lgbm", lgbm)],
            final_estimator=LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
            ),
            cv=_make_cv(),
            n_jobs=-1,
        )
    if model_type == "rf":
        return CalibratedClassifierCV(
            _build_rf(best_params.get("rf")), method="isotonic", cv=CV_FOLDS
        )
    if model_type == "xgb":
        return _build_xgb(best_params.get("xgb"))
    # lgbm
    return _build_lgbm(best_params.get("lgbm"))


def _make_final_model_b(model_type: str, best_params: dict) -> object:
    """Build Model B using best single-model type (Stacking too heavy for 90 samples)."""
    # if stacking won overall, fall back to LGBM for Model B
    effective = "lgbm" if model_type == "stack" else model_type
    if effective == "rf":
        return CalibratedClassifierCV(
            _build_rf(best_params.get("rf")), method="isotonic", cv=3
        )
    if effective == "xgb":
        return _build_xgb(best_params.get("xgb"))
    return _build_lgbm(best_params.get("lgbm"))


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    t0 = time.time()

    if not CATALOG_FILE.is_file():
        print(f"ERROR: catalog not found: {CATALOG_FILE}", file=sys.stderr)
        return 1

    FIGURES_DIR.mkdir(exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=== GASP Phase 09 v2: Optimized Taxonomy Classifier ===")
    print(f"Input      : {CATALOG_FILE}")
    print(f"Threshold  : {CONF_THRESHOLD}")
    print(f"Optuna     : {OPTUNA_N_TRIALS} trials × 3 models")
    print()

    # ── load & clean ──────────────────────────────────────────────────────────
    df = pd.read_parquet(CATALOG_FILE)
    df = df.drop(columns=[c for c in
                           ["complex", "taxonomy_ml", "taxonomy_ml_prob",
                            "taxonomy_ml_source", "taxonomy_final"]
                           if c in df.columns])
    n_total = len(df)
    print(f"Loaded : {n_total:,} asteroids, {df.shape[1]} columns")

    df["complex"] = df["taxonomy"].map(COMPLEX_MAP)
    lab_mask  = df["complex"].notna()
    n_labeled = int(lab_mask.sum())

    print(f"Labeled: {n_labeled}")
    vc = df.loc[lab_mask, "complex"].value_counts()
    for cls, cnt in vc.items():
        print(f"  {cls}: {cnt:3d}  ({cnt/n_labeled*100:.1f}%)")
    print()

    # ── impute Feature Set A (not albedo) ─────────────────────────────────────
    df_imp = _impute_median(df.copy(), FEATURE_COLS_A)

    # ── Feature Set A training arrays ─────────────────────────────────────────
    X_a    = df_imp.loc[lab_mask, FEATURE_COLS_A].values.astype(np.float32)
    y_a    = df.loc[lab_mask, "complex"].values
    le_a   = LabelEncoder().fit(y_a)
    y_enc_a = le_a.transform(y_a)
    cv_a   = _make_cv()

    # ── Feature Set B training arrays ─────────────────────────────────────────
    lab_b_all  = lab_mask & df["albedo"].notna()
    vc_b       = df.loc[lab_b_all, "complex"].value_counts()
    keep_b     = [cls for cls, n in vc_b.items() if n >= MIN_B_CLASS_N]
    excluded_b = [cls for cls, n in vc_b.items() if n < MIN_B_CLASS_N]
    lab_b_mask = lab_b_all & df["complex"].isin(keep_b)
    n_b_train  = int(lab_b_mask.sum())

    X_b_spec = df_imp.loc[lab_b_mask, FEATURE_COLS_A].values.astype(np.float32)
    X_b_alb  = df.loc[lab_b_mask, "albedo"].values.reshape(-1, 1).astype(np.float32)
    X_b      = np.hstack([X_b_spec, X_b_alb])
    y_b      = df.loc[lab_b_mask, "complex"].values
    le_b     = LabelEncoder().fit(y_b)
    y_enc_b  = le_b.transform(y_b)

    # ════════════════════════════════════════════════════════════════════════════
    # OPTUNA HYPERPARAMETER SEARCH  (Feature Set A)
    # ════════════════════════════════════════════════════════════════════════════
    print("─" * 60)
    print(f"Optuna search — Feature Set A  (n={n_labeled}, {OPTUNA_N_TRIALS} trials/model)")
    print("─" * 60)

    t_opt = time.time()

    print("  [1/3] RandomForest …")
    rf_params,   rf_best   = _run_optuna("RF",   _rf_objective(X_a, y_enc_a, cv_a),   OPTUNA_N_TRIALS)

    print("  [2/3] XGBoost …")
    xgb_params,  xgb_best  = _run_optuna("XGB",  _xgb_objective(X_a, y_enc_a, cv_a),  OPTUNA_N_TRIALS)

    print("  [3/3] LightGBM …")
    lgbm_params, lgbm_best = _run_optuna("LGBM", _lgbm_objective(X_a, y_enc_a, cv_a), OPTUNA_N_TRIALS)

    print(f"\nOptuna done in {time.time()-t_opt:.1f}s")
    print()

    # persist best params
    best_params = {"rf": rf_params, "xgb": xgb_params, "lgbm": lgbm_params}
    PARAMS_FILE.write_text(
        json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(),
                    "n_trials": OPTUNA_N_TRIALS,
                    **{k: {"best_f1_macro": round(v, 4), "params": p}
                       for k, (p, v) in [("rf",   (rf_params, rf_best)),
                                          ("xgb",  (xgb_params, xgb_best)),
                                          ("lgbm", (lgbm_params, lgbm_best))]}},
                   indent=2),
        encoding="utf-8",
    )
    print(f"Best params → {PARAMS_FILE}")
    print()

    # ════════════════════════════════════════════════════════════════════════════
    # CV EVALUATION — all 4 models  (Feature Set A)
    # ════════════════════════════════════════════════════════════════════════════
    print("─" * 60)
    print("CV evaluation — Feature Set A (5-fold, cross_val_predict)")
    print("─" * 60)

    results: dict[str, dict] = {}

    print("\n[RF]")
    results["rf"] = _eval_cv(
        _build_rf({**rf_params, "class_weight": "balanced",
                   "random_state": RANDOM_STATE, "n_jobs": -1}),
        X_a, y_enc_a, le_a, "RandomForest (tuned)", CM_A_FIGURE, plt,
    )

    print("[XGB]")
    results["xgb"] = _eval_cv(
        _build_xgb({**xgb_params, "random_state": RANDOM_STATE,
                    "n_jobs": -1, "verbosity": 0}),
        X_a, y_enc_a, le_a, "XGBoost (tuned)", None, plt,
    )

    print("[LGBM]")
    results["lgbm"] = _eval_cv(
        _build_lgbm({**lgbm_params, "class_weight": "balanced",
                     "random_state": RANDOM_STATE, "n_jobs": -1, "verbose": -1}),
        X_a, y_enc_a, le_a, "LightGBM (tuned)", None, plt,
    )

    print("[Stacking: RF + XGB + LGBM → LogisticRegression]")
    stack_model = StackingClassifier(
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
        cv=_make_cv(),
        n_jobs=-1,
    )
    results["stack"] = _eval_cv(
        stack_model, X_a, y_enc_a, le_a,
        "Stacking (RF+XGB+LGBM → LR)", None, plt,
    )

    # ── comparison table ───────────────────────────────────────────────────────
    print("─" * 60)
    print("Model comparison — Feature Set A")
    print("─" * 60)
    classes_a = list(le_a.classes_)
    hdr = f"{'Model':20s}  {'Acc':>6s}  {'F1-mac':>7s}  " + \
          "  ".join(f"F1-{c:>1s}" for c in classes_a)
    print(hdr)
    print("─" * len(hdr))
    for key, r in results.items():
        row = (f"{r['label'][:20]:20s}  {r['accuracy']*100:5.1f}%  "
               f"{r['f1_macro']:.4f}   " +
               "  ".join(f"{r['f1_class'].get(c, 0):.3f}" for c in classes_a))
        print(row)
    print()

    # ── pick winner ───────────────────────────────────────────────────────────
    best_type = max(results, key=lambda k: results[k]["f1_macro"])
    best_f1   = results[best_type]["f1_macro"]
    src_tag   = f"gasp_{best_type}_v2"
    print(f"Winner: {best_type.upper()}  (F1-macro={best_f1:.4f}  → source tag: {src_tag})")
    print()

    # ════════════════════════════════════════════════════════════════════════════
    # TRAIN FINAL MODELS (A + B)
    # ════════════════════════════════════════════════════════════════════════════
    print("─" * 60)
    print(f"Training final Model A ({best_type.upper()}) on all {n_labeled} labeled …")
    final_a = _make_final_model_a(best_type, best_params, results)
    final_a.fit(X_a, y_enc_a)
    print("Done.")

    print(f"\nTraining final Model B ({best_type.upper() if best_type!='stack' else 'LGBM'}) "
          f"on {n_b_train} labeled with albedo …")
    if excluded_b:
        print(f"  Excluded classes (< {MIN_B_CLASS_N} samples): {excluded_b}")
    final_b = _make_final_model_b(best_type, best_params)
    k_b = min(CV_FOLDS, int(pd.Series(y_enc_b).value_counts().min()))
    if hasattr(final_b, "cv"):                    # CalibratedClassifierCV
        final_b.cv = k_b
    final_b.fit(X_b, y_enc_b)
    print("Done.")
    print()

    # ── CV report for Model B ─────────────────────────────────────────────────
    print(f"CV evaluation — Model B  (n={n_b_train}, k={k_b})")
    eff_type_b = "lgbm" if best_type == "stack" else best_type
    eval_model_b = _make_final_model_b(best_type, best_params)
    cv_b_result = _eval_cv(
        eval_model_b, X_b, y_enc_b, le_b,
        f"Model B ({eff_type_b.upper()}+albedo)", CM_B_FIGURE, plt,
    )

    # X-type F1 comparison across all A models + B
    print("X-type F1 across models (Feature Set A → B):")
    for k, r in results.items():
        xf = r["f1_class"].get("X", "n/a")
        print(f"  {k:8s}: {xf:.3f}" if isinstance(xf, float) else f"  {k:8s}: {xf}")
    b_xf1 = cv_b_result["f1_class"].get("X", "n/a")
    print(f"  model_b : {b_xf1:.3f}" if isinstance(b_xf1, float) else f"  model_b : {b_xf1}")
    print()

    # ════════════════════════════════════════════════════════════════════════════
    # PREDICTIONS
    # ════════════════════════════════════════════════════════════════════════════
    print("Predicting for all 19,190 asteroids …")

    # Model A → all rows
    X_all_a  = df_imp[FEATURE_COLS_A].values.astype(np.float32)
    proba_a  = final_a.predict_proba(X_all_a)
    pred_a   = le_a.inverse_transform(np.argmax(proba_a, axis=1))
    prob_a   = proba_a.max(axis=1)

    # Model B → rows with real albedo
    has_alb  = df["albedo"].notna().values
    alb_vals = df["albedo"].values
    X_b_all  = np.hstack([
        df_imp.loc[has_alb, FEATURE_COLS_A].values.astype(np.float32),
        alb_vals[has_alb].reshape(-1, 1).astype(np.float32),
    ])
    proba_b  = final_b.predict_proba(X_b_all)
    pred_b   = le_b.inverse_transform(np.argmax(proba_b, axis=1))
    prob_b   = proba_b.max(axis=1)

    # X-subclassification on Model B X-predictions
    pred_b_ref = pred_b.copy()
    for i, (p, alb) in enumerate(zip(pred_b, alb_vals[has_alb])):
        if p == "X" and np.isfinite(alb) and alb > 0:
            pred_b_ref[i] = _x_subclass(float(alb))

    # Merge: default A, override B where albedo present
    taxonomy_ml  = pred_a.copy()
    tax_ml_prob  = prob_a.copy()
    tax_ml_src   = np.full(n_total, f"{src_tag}_A", dtype=object)

    alb_idx = np.where(has_alb)[0]
    taxonomy_ml[alb_idx] = pred_b_ref
    tax_ml_prob[alb_idx] = prob_b
    tax_ml_src[alb_idx]  = f"{src_tag}_B"

    df["taxonomy_ml"]        = taxonomy_ml
    df["taxonomy_ml_prob"]   = tax_ml_prob.round(4)
    df["taxonomy_ml_source"] = tax_ml_src

    # taxonomy_final
    mahlke_ok = df["taxonomy"].notna().values
    ml_ok     = tax_ml_prob >= CONF_THRESHOLD
    df["taxonomy_final"] = np.where(
        mahlke_ok, df["taxonomy"].values,
        np.where(ml_ok, taxonomy_ml, np.nan),
    )

    # ── coverage report ────────────────────────────────────────────────────────
    n_mahlke   = int(mahlke_ok.sum())
    n_ml_added = int((~mahlke_ok & ml_ok).sum())
    n_low_conf = int((~mahlke_ok & ~ml_ok).sum())
    n_final    = int(df["taxonomy_final"].notna().sum())
    n_use_b    = int((tax_ml_src == f"{src_tag}_B").sum())
    n_x_sub    = int(np.isin(taxonomy_ml[alb_idx], ["E", "M", "P"]).sum())

    print()
    print("=== Coverage Report ===")
    print(f"  Mahlke labels (original)     : {n_mahlke:,}")
    print(f"  ML added (prob ≥ {CONF_THRESHOLD})        : {n_ml_added:,}")
    print(f"  Low-confidence (excluded)    : {n_low_conf:,}")
    print(f"  taxonomy_final total          : {n_final:,}  "
          f"({n_final/n_total*100:.1f}% of {n_total:,})")
    print()
    print(f"  Model A used ({src_tag}_A) : {n_total-n_use_b:,}")
    print(f"  Model B used ({src_tag}_B) : {n_use_b:,}")
    print(f"    of which X→E/M/P refined  : {n_x_sub:,}")
    print()
    print("taxonomy_final distribution:")
    fc = df["taxonomy_final"].value_counts()
    for cls, cnt in fc.items():
        print(f"  {cls:4s}: {cnt:6,}  ({cnt/n_total*100:.1f}%)")
    print()

    # ── backup + write catalog ─────────────────────────────────────────────────
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
        "winner_model":         best_type,
        "source_tag":           src_tag,
        "optuna_n_trials":      OPTUNA_N_TRIALS,
        "confidence_threshold": CONF_THRESHOLD,
        "complex_map":          COMPLEX_MAP,
        "x_albedo_thresholds":  {"E_min": X_ALBEDO_E_MIN, "P_max": X_ALBEDO_P_MAX},
        "optuna_results": {
            k: {"best_f1_macro": round(v, 4), "params": p}
            for k, (p, v) in [("rf",  (rf_params,   rf_best)),
                               ("xgb", (xgb_params,  xgb_best)),
                               ("lgbm",(lgbm_params, lgbm_best))]
        },
        "cv_comparison": {
            k: {"f1_macro": r["f1_macro"], "accuracy": r["accuracy"],
                "f1_class": r["f1_class"]}
            for k, r in results.items()
        },
        "model_b": {
            "effective_type":   eff_type_b,
            "n_labeled":        int(n_b_train),
            "excluded_classes": excluded_b,
            "cv_f1_macro":      cv_b_result["f1_macro"],
            "cv_x_f1":          cv_b_result["f1_class"].get("X"),
        },
        "coverage": {
            "n_mahlke":          int(n_mahlke),
            "n_ml_added":        int(n_ml_added),
            "n_low_confidence":  int(n_low_conf),
            "n_taxonomy_final":  int(n_final),
            "taxonomy_final_pct": round(n_final / n_total * 100, 2),
            "n_x_subclassified": int(n_x_sub),
        },
        "output_file":    str(CATALOG_FILE),
        "backup_file":    str(bak),
        "params_file":    str(PARAMS_FILE),
        "cm_model_a":     str(CM_A_FIGURE),
        "cm_model_b":     str(CM_B_FIGURE),
        "duration_seconds": round(time.time() - t0, 1),
    }
    LOG_FILE.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"Log    : {LOG_FILE}")
    print()
    print(f"Done in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
