#!/usr/bin/env python3
"""
10_taxonomy_retrain.py
GASP — RF taxonomy retrain with augmented labels (Mahlke + PDS Bus-DeMeo).

enrich_external.py added spectral_class_best (PDS Bus-DeMeo/Bus/Tholen) for
392 objects in GASP v2. Of these, ~275 have no Mahlke label but have full
reflectance features — they can augment the RF training set.

Strategy:
  1. Map PDS spectral sub-classes to the same broad complex scheme as Mahlke
     (S / C / X / V / D) using a conservative label map (skip uncertain,
     mixed-type, or ambiguous labels).
  2. Combine with Mahlke labels (priority: Mahlke > PDS). Skip if conflicting.
  3. Re-train CalibratedRF using the saved Optuna hyperparameters (no re-tuning).
  4. Evaluate improvement via 5-fold cross_val_predict.
  5. Write updated taxonomy columns to GASP v2 → v2 (in-place update).

Outputs:
  data/final/gasp_catalog_v2.parquet  (updated taxonomy_ml, taxonomy_final)
  data/final/10_retrain_log.json
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[1]
FINAL_DIR    = ROOT / "data" / "final"
V2_PATH      = FINAL_DIR / "gasp_catalog_v2.parquet"
PARAMS_FILE  = FINAL_DIR / "optuna_best_params.json"
LOG_FILE     = FINAL_DIR / "10_retrain_log.json"

# ── features (same as 09_taxonomy_classifier.py) ──────────────────────────────
BANDS        = [374, 418, 462, 506, 550, 594, 638, 682,
                726, 770, 814, 858, 902, 946, 990, 1034]
REFL_COLS    = [f"refl_{b}" for b in BANDS]
SLOPE_COLS   = ["s1", "s2", "s3", "s4"]
FEATURE_COLS = REFL_COLS + SLOPE_COLS  # 20 features

CONF_THRESHOLD = 0.70
CV_FOLDS       = 5
RANDOM_STATE   = 42
SOURCE_TAG     = "gasp_rf_v3"

# ── X-complex albedo thresholds ────────────────────────────────────────────────
X_ALBEDO_E_MIN = 0.30
X_ALBEDO_P_MAX = 0.10

# ── PDS sub-class → broad complex map ─────────────────────────────────────────
# Conservative: only unambiguous single-complex labels are mapped.
# Mixed types (XC, CX, DX…), uncertain suffixes (:), survey flags (U) → None.
PDS_COMPLEX_MAP: dict[str, str] = {
    # S-complex
    "S": "S", "Sa": "S", "Sk": "S", "Sl": "S", "Sq": "S",
    "Sqw": "S", "Sr": "S", "Srw": "S", "Sw": "S",
    "A": "S", "K": "S", "L": "S", "Ld": "S", "Q": "S",
    # C-complex
    "C": "C", "Cb": "C", "Cgh": "C", "Ch": "C", "B": "C", "F": "C",
    # X-complex (includes E, M, P sub-types — mapped to X; albedo rule applied post-hoc)
    "X": "X", "Xe": "X", "Xk": "X", "Xc": "X",
    "M": "X", "E": "X", "P": "X",
    # V-type
    "V": "V",
    # D-complex
    "D": "D", "DT": "D",
    # Explicitly excluded (returned as None):
    # T, O, CX, XC, DX, SD, SC, ST, SG, SU, XFU, BCU, PDC, CGU, FX, etc.
}

# ── Mahlke complex map (same as 09_taxonomy_classifier.py) ────────────────────
MAHLKE_COMPLEX_MAP: dict[str, str] = {
    "S": "S", "A": "S", "K": "S", "L": "S", "Q": "S",
    "C": "C", "B": "C", "Ch": "C",
    "X": "X", "Xe": "X", "Xk": "X", "M": "X", "Mk": "X", "P": "X", "Pk": "X",
    "V": "V",
    "D": "D", "Z": "D",
}


def map_pds(label: str) -> str | None:
    """Map a PDS spectral label to a broad complex. Returns None if ambiguous."""
    if not isinstance(label, str) or label.strip() in ("", "nan", "***"):
        return None
    label = label.strip()
    # Reject uncertain (:) and multi-type labels containing '/'
    if ":" in label or "/" in label:
        return None
    return PDS_COMPLEX_MAP.get(label, None)


def _x_subclass(albedo: float) -> str:
    if albedo > X_ALBEDO_E_MIN:
        return "E"
    if albedo < X_ALBEDO_P_MAX:
        return "P"
    return "M"


def _impute_median(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())
    return df


def _build_rf(params: dict) -> RandomForestClassifier:
    return RandomForestClassifier(
        **{k: v for k, v in params.items()
           if k in ("n_estimators", "max_depth", "min_samples_split",
                    "min_samples_leaf", "max_features", "class_weight",
                    "random_state", "n_jobs")},
    )


def main() -> None:
    print("\n" + "=" * 65)
    print("  GASP Step 10 — RF taxonomy retrain (Mahlke + PDS augment)")
    print("=" * 65)

    if not V2_PATH.exists():
        print(f"  ERROR: {V2_PATH} not found"); return
    if not PARAMS_FILE.exists():
        print(f"  ERROR: {PARAMS_FILE} not found — run 09_taxonomy_classifier.py first")
        return

    df = pd.read_parquet(V2_PATH)
    print(f"\n  GASP v2: {len(df):,} rows, {len(df.columns)} cols")

    # ── impute median reflectance ─────────────────────────────────────────────
    df_imp = _impute_median(df.copy(), FEATURE_COLS)

    # ── build augmented label set ─────────────────────────────────────────────
    labels = pd.Series(np.nan, index=df.index, dtype=object)

    # Priority 1: Mahlke 2022 (same as original 09_taxonomy_classifier.py)
    mahlke_ok = df["taxonomy"].notna()
    for idx in df.index[mahlke_ok]:
        raw = str(df.at[idx, "taxonomy"])
        mapped = MAHLKE_COMPLEX_MAP.get(raw, None)
        if mapped:
            labels[idx] = mapped

    n_mahlke = labels.notna().sum()
    print(f"\n  Mahlke 2022 labels (mapped):  {n_mahlke:,}")

    # Priority 2: PDS spectral labels (only for objects without Mahlke label)
    if "spectral_class_best" in df.columns:
        for idx in df.index[df["taxonomy"].isna() & df["spectral_class_best"].notna()]:
            raw    = str(df.at[idx, "spectral_class_best"])
            mapped = map_pds(raw)
            if mapped:
                labels[idx] = mapped

    n_pds   = labels.notna().sum() - n_mahlke
    n_total_labeled = labels.notna().sum()
    print(f"  PDS spectral labels (added):  {n_pds:,}")
    print(f"  Total labeled (training set): {n_total_labeled:,}  "
          f"(was {n_mahlke:,}, +{n_pds:,} = "
          f"+{n_pds/n_mahlke*100:.1f}%)")

    # Restrict to rows with valid features and label
    feat_ok = df_imp[FEATURE_COLS].notna().all(axis=1)
    labeled_mask = labels.notna() & feat_ok
    n_labeled = labeled_mask.sum()
    print(f"  With valid features:          {n_labeled:,}")

    X = df_imp.loc[labeled_mask, FEATURE_COLS].values.astype(np.float32)
    y_raw = labels[labeled_mask].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    classes = list(le.classes_)
    print(f"\n  Classes: {classes}")
    for cls in classes:
        n_cls = (y_raw == cls).sum()
        print(f"    {cls}: {n_cls:,}")

    # ── Load saved Optuna params ──────────────────────────────────────────────
    saved = json.loads(PARAMS_FILE.read_text())
    rf_params = saved["rf"]["params"]
    print(f"\n  Loaded Optuna params (best RF F1-macro was "
          f"{saved['rf']['best_f1_macro']:.4f})")

    # ── CV evaluation (old vs new training set) ───────────────────────────────
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rf_est = _build_rf({**rf_params, "class_weight": "balanced",
                        "random_state": RANDOM_STATE, "n_jobs": -1})

    print(f"\n  5-fold CV on augmented training set (n={n_labeled:,}):")
    y_pred_cv = cross_val_predict(rf_est, X, y_enc, cv=cv)
    y_pred_cv_labels = le.inverse_transform(y_pred_cv)
    f1_new = f1_score(y_raw, y_pred_cv_labels, average="macro")
    print(classification_report(y_raw, y_pred_cv_labels, target_names=classes))
    print(f"  F1-macro (augmented): {f1_new:.4f}")

    # Compare with original (Mahlke-only) subset
    mahlke_only_mask = labeled_mask & mahlke_ok & feat_ok
    n_mahlke_feat = mahlke_only_mask.sum()
    X_m = df_imp.loc[mahlke_only_mask, FEATURE_COLS].values.astype(np.float32)
    y_m = labels[mahlke_only_mask].values
    le_m = LabelEncoder(); y_m_enc = le_m.fit_transform(y_m)
    y_pred_m = cross_val_predict(
        _build_rf({**rf_params, "class_weight": "balanced",
                   "random_state": RANDOM_STATE, "n_jobs": -1}),
        X_m, y_m_enc, cv=cv
    )
    f1_orig = f1_score(y_m, le_m.inverse_transform(y_pred_m), average="macro")
    print(f"  F1-macro (Mahlke-only baseline, n={n_mahlke_feat:,}): {f1_orig:.4f}")
    print(f"  Improvement: {(f1_new - f1_orig)*100:+.2f} pp")

    # ── Train final model on full augmented set ───────────────────────────────
    print(f"\n  Training final CalibratedRF on {n_labeled:,} labeled objects …")
    final_rf = CalibratedClassifierCV(
        _build_rf({**rf_params, "class_weight": "balanced",
                   "random_state": RANDOM_STATE, "n_jobs": -1}),
        method="isotonic", cv=CV_FOLDS,
    )
    final_rf.fit(X, y_enc)

    # ── Predict for all 19,190 ────────────────────────────────────────────────
    print("  Predicting for all asteroids …")
    X_all = df_imp[FEATURE_COLS].values.astype(np.float32)
    proba = final_rf.predict_proba(X_all)
    pred  = le.inverse_transform(np.argmax(proba, axis=1))
    prob  = proba.max(axis=1)

    # X→E/M/P post-hoc refinement
    alb_col = "albedo"
    alb_vals = df[alb_col].values if alb_col in df.columns else np.full(len(df), np.nan)
    x_refined = np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        if pred[i] == "X" and np.isfinite(alb_vals[i]) and alb_vals[i] > 0:
            pred[i] = _x_subclass(float(alb_vals[i]))
            x_refined[i] = True

    df["taxonomy_ml"]           = pred
    df["taxonomy_ml_prob"]      = prob.round(4)
    df["taxonomy_ml_source"]    = SOURCE_TAG
    df["taxonomy_ml_x_refined"] = x_refined

    # taxonomy_final: Mahlke > PDS spectral > ML (conf >= threshold)
    has_mahlke = df["taxonomy"].notna()
    has_pds    = (df.get("spectral_class_best", pd.Series(dtype=str)).notna() &
                  ~has_mahlke)
    ml_ok_flag = prob >= CONF_THRESHOLD

    taxonomy_final = np.full(len(df), np.nan, dtype=object)
    taxonomy_final[has_mahlke.values] = df.loc[has_mahlke, "taxonomy"].values
    for idx in df.index[has_pds]:
        mapped = map_pds(str(df.at[idx, "spectral_class_best"]))
        if mapped:
            taxonomy_final[idx] = mapped
    remaining = np.isnan(taxonomy_final.astype(str))
    remaining = pd.Series(taxonomy_final).isna().values & ~has_mahlke.values & ~has_pds.values
    taxonomy_final[remaining & ml_ok_flag] = pred[remaining & ml_ok_flag]
    df["taxonomy_final"] = taxonomy_final

    n_final = pd.notna(taxonomy_final).sum()
    print(f"\n  taxonomy_final: {n_final:,} / {len(df):,} "
          f"({n_final/len(df)*100:.1f}%)")
    print("  Distribution:")
    for cls, cnt in df["taxonomy_final"].value_counts().items():
        print(f"    {cls}: {cnt:,}")

    df.to_parquet(V2_PATH, index=False)
    print(f"\n  → GASP v2 updated in-place: {V2_PATH}")

    log = {
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "n_mahlke":        int(n_mahlke),
        "n_pds_added":     int(n_pds),
        "n_labeled_total": int(n_labeled),
        "f1_macro_orig":   round(f1_orig, 4),
        "f1_macro_new":    round(f1_new, 4),
        "improvement_pp":  round((f1_new - f1_orig) * 100, 2),
        "classes":         classes,
        "source_tag":      SOURCE_TAG,
    }
    LOG_FILE.write_text(json.dumps(log, indent=2))
    print(f"  Log → {LOG_FILE}\n")


if __name__ == "__main__":
    main()
