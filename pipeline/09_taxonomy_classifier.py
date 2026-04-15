#!/usr/bin/env python3
"""GASP Phase 09: Dual-model taxonomy classifier with albedo-gated X-subclassification.

Two Random Forest models are trained on Mahlke et al. (2022) labeled asteroids:

  Model A — Feature Set A (16 Gaia bands + s1–s4, 20 features)
    Training set : all 358 labeled asteroids
    Coverage     : all 19,190 (refl_418–990 = 100 %, refl_374/s1 imputed)

  Model B — Feature Set B (Feature Set A + albedo, 21 features)
    Training set : labeled asteroids with NEOWISE albedo (≥ 5 samples/class)
    Coverage     : ~22.3 % of catalog (4,279 asteroids with albedo)
    Note         : D-type excluded from Model B (only 1 training sample with albedo)

Prediction routing:
  • Asteroid has albedo → use Model B prediction + prob
  • Asteroid lacks albedo → use Model A prediction + prob
  • Model B predicts X AND albedo is available → refine to E / M / P
      E : albedo > 0.30   (enstatite-like, Tholen 1984)
      M : 0.10 ≤ albedo ≤ 0.30  (metallic)
      P : albedo < 0.10   (primitive, low-albedo)

Output columns (added/overwritten in catalog):
  taxonomy_ml        — predicted class (broad complex or E/M/P for X-subtypes)
  taxonomy_ml_prob   — max class probability from the selected model
  taxonomy_ml_source — "gasp_rf_v1_A" or "gasp_rf_v1_B"
  taxonomy_final     — Mahlke label if available, else ML prediction (prob ≥ 0.70)

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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
FINAL_DIR   = Path("data/final")
FIGURES_DIR = Path("figures")

CATALOG_FILE = FINAL_DIR / "gasp_catalog_v1.parquet"
LOG_FILE     = FINAL_DIR / "09_taxonomy_classifier_log.json"
CM_A_FIGURE  = FIGURES_DIR / "09_cm_model_a.png"
CM_B_FIGURE  = FIGURES_DIR / "09_cm_model_b.png"

# ── feature sets ───────────────────────────────────────────────────────────────
BANDS          = [374, 418, 462, 506, 550, 594, 638, 682,
                  726, 770, 814, 858, 902, 946, 990, 1034]
REFL_COLS      = [f"refl_{b}" for b in BANDS]
SLOPE_COLS     = ["s1", "s2", "s3", "s4"]
FEATURE_COLS_A = REFL_COLS + SLOPE_COLS               # 20 features
FEATURE_COLS_B = REFL_COLS + SLOPE_COLS + ["albedo"]  # 21 features

# ── X-complex albedo thresholds for subclassification (after Tholen 1984) ─────
#   E-type : pv > 0.30  (enstatite-like, high albedo)
#   M-type : 0.10 ≤ pv ≤ 0.30  (metallic)
#   P-type : pv < 0.10  (primitive, low albedo)
X_ALBEDO_E_MIN = 0.30
X_ALBEDO_P_MAX = 0.10

# ── training / model hyper-parameters ─────────────────────────────────────────
CONF_THRESHOLD  = 0.70
CV_FOLDS        = 5
RF_N_ESTIMATORS = 500
RANDOM_STATE    = 42
MIN_B_CLASS_N   = 5   # minimum samples per class to include in Model B training

# ── broad-complex mapping (Mahlke et al. 2022) ────────────────────────────────
# S-complex : S, A (olivine-rich), K, L, Q (OC-like)
# C-complex : C, B, Ch (hydrated)
# X-complex : X, Xe, Xk, M (metallic), Mk, P (low-albedo), Pk
# V         : V (basaltic, Vesta-family)
# D         : D (primitive/red), Z (rare primitive)
COMPLEX_MAP: dict[str, str] = {
    "S":  "S", "A":  "S", "K":  "S", "L":  "S", "Q":  "S",
    "C":  "C", "B":  "C", "Ch": "C",
    "X":  "X", "Xe": "X", "Xk": "X", "M":  "X", "Mk": "X", "P":  "X", "Pk": "X",
    "V":  "V",
    "D":  "D", "Z":  "D",
}


# ── helpers ────────────────────────────────────────────────────────────────────

def _impute_median(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Fill NaNs with per-column global median (modifies df in place)."""
    for c in cols:
        if c in df.columns:
            med = df[c].median()
            df[c] = df[c].fillna(med)
    return df


def _build_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def _x_subclass(albedo: float) -> str:
    """Assign E / M / P sub-label to an X-complex asteroid based on albedo."""
    if albedo >= X_ALBEDO_E_MIN:
        return "E"
    if albedo < X_ALBEDO_P_MAX:
        return "P"
    return "M"


def _run_cv(
    X: np.ndarray,
    y_enc: np.ndarray,
    le: LabelEncoder,
    label: str,
    cm_path: Path,
    plt_mod,
) -> dict:
    """Run StratifiedKFold CV, print report, save confusion matrix.

    Automatically reduces k when the smallest class has fewer than CV_FOLDS samples.
    Returns a metrics dict suitable for the JSON log.
    """
    classes = le.classes_
    counts  = pd.Series(y_enc).value_counts()
    min_n   = int(counts.min())
    k       = min(CV_FOLDS, min_n)

    if k < CV_FOLDS:
        print(f"  [Note] Reducing CV folds from {CV_FOLDS} → {k} "
              f"(smallest class has {min_n} sample{'s' if min_n != 1 else ''})")
    if k < 2:
        print(f"  [Warning] Cannot run CV for {label}: need ≥ 2 samples per class.")
        return {"n": int(len(y_enc)), "k_folds": 0,
                "accuracy": None, "f1_macro": None, "x_f1": None}

    cv        = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    y_pred    = cross_val_predict(_build_rf(), X, y_enc, cv=cv, n_jobs=-1)
    y_labels  = le.inverse_transform(y_enc)
    yp_labels = le.inverse_transform(y_pred)

    acc    = float((y_pred == y_enc).mean())
    f1_mac = float(f1_score(y_enc, y_pred, average="macro"))

    # Per-class F1 for X specifically (if present in this model)
    x_f1 = None
    if "X" in classes:
        x_idx = int(np.where(classes == "X")[0][0])
        x_f1  = float(f1_score(y_enc, y_pred, labels=[x_idx], average="macro"))

    print(f"=== CV Results — {label}  (n={len(y_enc)}, k={k}) ===")
    print(f"  Accuracy (macro)  : {acc*100:.1f}%")
    print(f"  F1-macro          : {f1_mac:.3f}")
    if x_f1 is not None:
        print(f"  X-type F1         : {x_f1:.3f}")
    print()
    report_str = classification_report(
        y_labels, yp_labels, target_names=classes, zero_division=0
    )
    print("  Classification report:")
    for line in report_str.splitlines():
        print(f"    {line}")

    # Confusion matrix figure
    fig, ax = plt_mod.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_labels, yp_labels,
        display_labels=classes,
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(
        f"Taxonomy RF — {label} — {k}-fold CV\n"
        f"Acc={acc*100:.1f}%  F1-macro={f1_mac:.3f}  (n={len(y_enc)})",
        fontsize=11,
    )
    plt_mod.tight_layout()
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt_mod.close(fig)
    print(f"  Confusion matrix → {cm_path}")
    print()

    return {
        "n":        int(len(y_enc)),
        "k_folds":  int(k),
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_mac, 4),
        "x_f1":     round(x_f1, 4) if x_f1 is not None else None,
    }


def _train_calibrated(
    X: np.ndarray,
    y_enc: np.ndarray,
    k: int,
) -> CalibratedClassifierCV:
    """Train a calibrated RF on the full provided dataset."""
    model = CalibratedClassifierCV(_build_rf(), method="isotonic", cv=k)
    model.fit(X, y_enc)
    return model


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

    print("=== GASP Phase 09: Taxonomy Classifier (dual-model) ===")
    print(f"Input  : {CATALOG_FILE}")
    print(f"Threshold: {CONF_THRESHOLD}")
    print()

    # ── load & annotate ────────────────────────────────────────────────────────
    df = pd.read_parquet(CATALOG_FILE)

    # drop ML columns if this is a re-run
    drop_cols = ["complex", "taxonomy_ml", "taxonomy_ml_prob",
                 "taxonomy_ml_source", "taxonomy_final"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    n_total = len(df)
    print(f"Loaded : {n_total:,} asteroids, {df.shape[1]} columns")

    df["complex"] = df["taxonomy"].map(COMPLEX_MAP)
    lab_mask   = df["complex"].notna()
    n_labeled  = int(lab_mask.sum())

    print(f"Labeled (mappable taxonomy) : {n_labeled}")
    print()
    print("Complex distribution — all labeled:")
    vc_all = df.loc[lab_mask, "complex"].value_counts()
    for cls, cnt in vc_all.items():
        print(f"  {cls}: {cnt:3d}  ({cnt/n_labeled*100:.1f}%)")
    print()

    # ── feature checks ─────────────────────────────────────────────────────────
    for c in FEATURE_COLS_A:
        if c not in df.columns:
            print(f"ERROR: missing column: {c}", file=sys.stderr)
            return 1

    print("Feature completeness (all 19,190):")
    for c in FEATURE_COLS_B:
        cov = df[c].notna().mean()
        print(f"  {c:12s}: {cov*100:.1f}%")
    print()

    # ── impute Set-A features only (albedo intentionally not imputed) ──────────
    df_imp = df.copy()
    df_imp = _impute_median(df_imp, FEATURE_COLS_A)
    # albedo column keeps its NaNs — Model B runs only on rows where albedo is real

    # ═══════════════════════════════════════════════════════════════════════════
    # MODEL A — Feature Set A (all labeled, 5 classes)
    # ═══════════════════════════════════════════════════════════════════════════
    print("─" * 60)
    print("MODEL A — Feature Set A (Gaia bands + slopes, no albedo)")
    print("─" * 60)

    X_a    = df_imp.loc[lab_mask, FEATURE_COLS_A].values.astype(np.float32)
    y_a    = df.loc[lab_mask, "complex"].values
    le_a   = LabelEncoder().fit(y_a)
    y_enc_a = le_a.transform(y_a)

    print(f"Training set : {len(X_a)} asteroids, {len(FEATURE_COLS_A)} features")
    print(f"Classes      : {list(le_a.classes_)}")
    print()

    cv_a = _run_cv(X_a, y_enc_a, le_a, "Model A", CM_A_FIGURE, plt)

    print("Training calibrated Model A on full labeled set…")
    rf_a = _train_calibrated(X_a, y_enc_a, CV_FOLDS)
    print("Done.")
    print()

    # ═══════════════════════════════════════════════════════════════════════════
    # MODEL B — Feature Set B (labeled with albedo, classes ≥ MIN_B_CLASS_N)
    # ═══════════════════════════════════════════════════════════════════════════
    print("─" * 60)
    print("MODEL B — Feature Set B (Gaia bands + slopes + albedo)")
    print("─" * 60)

    lab_b_mask  = lab_mask & df["albedo"].notna()
    n_labeled_b = int(lab_b_mask.sum())
    print(f"Labeled with albedo  : {n_labeled_b}")

    vc_b = df.loc[lab_b_mask, "complex"].value_counts()
    print("Complex distribution — labeled with albedo:")
    for cls, cnt in vc_b.items():
        flag = "" if cnt >= MIN_B_CLASS_N else f"  ← excluded (< {MIN_B_CLASS_N})"
        print(f"  {cls}: {cnt:3d}{flag}")

    excluded_b = [cls for cls, cnt in vc_b.items() if cnt < MIN_B_CLASS_N]
    if excluded_b:
        print(f"\n  Classes excluded from Model B: {excluded_b}")
        print(f"  (insufficient training data; Model A handles these)")
    print()

    # filter out excluded classes
    keep_b_mask = lab_b_mask & df["complex"].isin(
        [cls for cls, cnt in vc_b.items() if cnt >= MIN_B_CLASS_N]
    )
    n_b_train = int(keep_b_mask.sum())

    # For Model B rows: use imputed spectral features but REAL albedo values
    X_b    = df_imp.loc[keep_b_mask, FEATURE_COLS_A].values.astype(np.float32)
    alb_b  = df.loc[keep_b_mask, "albedo"].values.reshape(-1, 1).astype(np.float32)
    X_b    = np.hstack([X_b, alb_b])

    y_b     = df.loc[keep_b_mask, "complex"].values
    le_b    = LabelEncoder().fit(y_b)
    y_enc_b = le_b.transform(y_b)

    print(f"Training set : {n_b_train} asteroids, {len(FEATURE_COLS_B)} features")
    print(f"Classes      : {list(le_b.classes_)}")
    print()

    cv_b = _run_cv(X_b, y_enc_b, le_b, "Model B", CM_B_FIGURE, plt)

    print("Training calibrated Model B on full labeled-with-albedo set…")
    k_b_train = min(CV_FOLDS, int(pd.Series(y_enc_b).value_counts().min()))
    rf_b = _train_calibrated(X_b, y_enc_b, k_b_train)
    print("Done.")
    print()

    # ── X-type F1 comparison ───────────────────────────────────────────────────
    print("─" * 60)
    print("X-type F1 comparison:")
    print(f"  Model A : {cv_a['x_f1']:.3f}")
    print(f"  Model B : {cv_b['x_f1']:.3f}"
          if cv_b["x_f1"] is not None
          else "  Model B : n/a (X not in classes)")
    if cv_b["x_f1"] is not None and cv_a["x_f1"] is not None:
        delta = cv_b["x_f1"] - cv_a["x_f1"]
        sign  = "+" if delta >= 0 else ""
        print(f"  Delta   : {sign}{delta:.3f}")
    print("─" * 60)
    print()

    # ═══════════════════════════════════════════════════════════════════════════
    # PREDICTIONS
    # ═══════════════════════════════════════════════════════════════════════════
    print("Predicting for all 19,190 asteroids…")

    # Model A → all rows
    X_all_a  = df_imp[FEATURE_COLS_A].values.astype(np.float32)
    proba_a  = rf_a.predict_proba(X_all_a)
    pred_a   = le_a.inverse_transform(np.argmax(proba_a, axis=1))
    prob_a   = proba_a.max(axis=1)

    # Model B → rows with real albedo
    has_albedo = df["albedo"].notna().values
    alb_vals   = df["albedo"].values   # NaN where absent

    X_spec_b  = df_imp.loc[has_albedo, FEATURE_COLS_A].values.astype(np.float32)
    alb_real  = alb_vals[has_albedo].reshape(-1, 1).astype(np.float32)
    X_all_b   = np.hstack([X_spec_b, alb_real])

    proba_b   = rf_b.predict_proba(X_all_b)
    pred_b    = le_b.inverse_transform(np.argmax(proba_b, axis=1))
    prob_b    = proba_b.max(axis=1)

    # Apply X-subclassification to Model B X-predictions
    pred_b_refined = pred_b.copy()
    for i, (p, alb) in enumerate(zip(pred_b, alb_vals[has_albedo])):
        if p == "X" and np.isfinite(alb) and alb > 0:
            pred_b_refined[i] = _x_subclass(float(alb))

    # Combine: default to Model A, override with Model B where albedo present
    taxonomy_ml     = pred_a.copy()
    taxonomy_ml_prob = prob_a.copy()
    taxonomy_ml_src  = np.array(["gasp_rf_v1_A"] * n_total, dtype=object)

    albedo_idx = np.where(has_albedo)[0]
    taxonomy_ml[albedo_idx]      = pred_b_refined
    taxonomy_ml_prob[albedo_idx] = prob_b
    taxonomy_ml_src[albedo_idx]  = "gasp_rf_v1_B"

    df["taxonomy_ml"]        = taxonomy_ml
    df["taxonomy_ml_prob"]   = taxonomy_ml_prob.round(4)
    df["taxonomy_ml_source"] = taxonomy_ml_src

    # ── taxonomy_final ─────────────────────────────────────────────────────────
    mahlke_ok = df["taxonomy"].notna().values
    ml_ok     = taxonomy_ml_prob >= CONF_THRESHOLD

    taxonomy_final = np.where(
        mahlke_ok,
        df["taxonomy"].values,
        np.where(ml_ok, taxonomy_ml, np.nan),
    )
    df["taxonomy_final"] = taxonomy_final

    # ── coverage report ────────────────────────────────────────────────────────
    n_mahlke   = int(mahlke_ok.sum())
    n_ml_added = int((~mahlke_ok & ml_ok).sum())
    n_low_conf = int((~mahlke_ok & ~ml_ok).sum())
    n_final    = int(df["taxonomy_final"].notna().sum())

    n_use_b    = int((taxonomy_ml_src == "gasp_rf_v1_B").sum())
    n_use_a    = int((taxonomy_ml_src == "gasp_rf_v1_A").sum())
    n_x_sub    = int(np.isin(taxonomy_ml[albedo_idx], ["E", "M", "P"]).sum())

    print()
    print("=== Coverage Report ===")
    print(f"  Mahlke labels (original)      : {n_mahlke:,}")
    print(f"  ML added (prob ≥ {CONF_THRESHOLD})         : {n_ml_added:,}")
    print(f"  Low-confidence (excluded)     : {n_low_conf:,}")
    print(f"  taxonomy_final total           : {n_final:,}  "
          f"({n_final/n_total*100:.1f}% of {n_total:,})")
    print()
    print(f"  Model A used (no albedo)      : {n_use_a:,}")
    print(f"  Model B used (has albedo)     : {n_use_b:,}")
    print(f"    of which X→E/M/P refined    : {n_x_sub:,}")
    print()
    print("taxonomy_final value distribution:")
    fc = df["taxonomy_final"].value_counts()
    for cls, cnt in fc.items():
        print(f"  {cls:4s}: {cnt:6,}  ({cnt/n_total*100:.1f}%)")
    print()

    # ── backup + write catalog ─────────────────────────────────────────────────
    backup_path = CATALOG_FILE.with_suffix(".parquet.bak")
    shutil.copy2(CATALOG_FILE, backup_path)
    print(f"Backup : {backup_path}")

    df = df.drop(columns=["complex"])
    df.to_parquet(CATALOG_FILE, index=False, compression="snappy")
    print(f"Saved  : {CATALOG_FILE}  ({df.shape[0]:,} rows × {df.shape[1]} cols)")

    # ── JSON log ───────────────────────────────────────────────────────────────
    duration = time.time() - t0
    log = {
        "timestamp":             datetime.now(timezone.utc).isoformat(),
        "gasp_phase":            "09_taxonomy_classifier",
        "model":                 "CalibratedClassifierCV(RandomForestClassifier)",
        "n_estimators":          RF_N_ESTIMATORS,
        "confidence_threshold":  CONF_THRESHOLD,
        "random_state":          RANDOM_STATE,
        "complex_map":           COMPLEX_MAP,
        "x_albedo_thresholds":   {
            "E_min": X_ALBEDO_E_MIN,
            "P_max": X_ALBEDO_P_MAX,
        },
        "model_a": {
            "feature_set":   "A",
            "feature_cols":  FEATURE_COLS_A,
            "n_features":    len(FEATURE_COLS_A),
            "n_labeled":     int(len(X_a)),
            "classes":       list(le_a.classes_),
            "cv_folds":      cv_a["k_folds"],
            "cv_accuracy":   cv_a["accuracy"],
            "cv_f1_macro":   cv_a["f1_macro"],
            "cv_x_f1":       cv_a["x_f1"],
        },
        "model_b": {
            "feature_set":       "B",
            "feature_cols":      FEATURE_COLS_B,
            "n_features":        len(FEATURE_COLS_B),
            "n_labeled":         int(n_b_train),
            "n_labeled_raw":     int(n_labeled_b),
            "classes":           list(le_b.classes_),
            "excluded_classes":  excluded_b,
            "min_class_n":       MIN_B_CLASS_N,
            "cv_folds":          cv_b["k_folds"],
            "cv_accuracy":       cv_b["accuracy"],
            "cv_f1_macro":       cv_b["f1_macro"],
            "cv_x_f1":           cv_b["x_f1"],
        },
        "predictions": {
            "n_total":           int(n_total),
            "n_using_model_a":   int(n_use_a),
            "n_using_model_b":   int(n_use_b),
            "n_x_subclassified": int(n_x_sub),
        },
        "coverage": {
            "n_mahlke":          int(n_mahlke),
            "n_ml_added":        int(n_ml_added),
            "n_low_confidence":  int(n_low_conf),
            "n_taxonomy_final":  int(n_final),
            "taxonomy_final_pct": round(n_final / n_total * 100, 2),
        },
        "output_file":           str(CATALOG_FILE),
        "backup_file":           str(backup_path),
        "cm_model_a":            str(CM_A_FIGURE),
        "cm_model_b":            str(CM_B_FIGURE),
        "duration_seconds":      round(time.time() - t0, 3),
    }
    LOG_FILE.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"Log    : {LOG_FILE}")
    print()
    print(f"Done in {time.time() - t0:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
