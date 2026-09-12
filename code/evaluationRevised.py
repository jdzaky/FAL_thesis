#!/usr/bin/env python3
"""
evaluation_three_models.py
Revisi evaluationRevised.py — tambah Decision Tree dan SVM
sebagai model tambahan selain Random Forest.

Perubahan dari versi sebelumnya:
  - Paired t-test dijalankan untuk 3 model: RF, DT, SVM
  - evaluate_descriptive() diperluas untuk 3 model sekaligus
  - Output JSON berisi hasil per model
  - Ringkasan tabel tesis menampilkan semua 3 model
"""

import os, sys, json, logging, warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import traceback

warnings.filterwarnings('ignore')

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASET_A = os.path.join(DATA_DIR, 'final_sdn_dataset_ahp.csv')
DATASET_B = os.path.join(DATA_DIR, 'GROUND_TRUTH.csv')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================
# KONFIGURASI PARAMETER
# ==============================================================
RANDOM_STATE = 42
N_ESTIMATORS = 100
N_SPLITS     = 5
N_JOBS       = -1
HOLDOUT_RATIO = 0.20  # dead code — dibiarkan dari versi sebelumnya

FEATURE_COLS = [
    'throughput_mbps', 'latency_ms', 'packet_loss_percent',
    'jitter_ms', 'cpu_usage_percent', 'memory_usage_mb',
    'flow_setup_time_ms', 'nodes'
]
LABEL_COL = 'best_controller_label'

# Definisi 3 model — tambahkan DT dan SVM di sini
MODELS = {
    'Random Forest': RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS
    ),
    'Decision Tree': DecisionTreeClassifier(
        random_state=RANDOM_STATE
    ),
    'SVM': SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        random_state=RANDOM_STATE
    )
}


# ==============================================================
# DEAD CODE — dibiarkan dari versi sebelumnya
# ==============================================================

def create_fixed_holdout(df_a, holdout_ratio=HOLDOUT_RATIO, random_state=RANDOM_STATE):
    le = LabelEncoder()
    y = le.fit_transform(df_a[LABEL_COL])
    idx_train, idx_holdout = train_test_split(
        np.arange(len(df_a)), test_size=holdout_ratio,
        stratify=y, random_state=random_state
    )
    df_train_pool = df_a.iloc[idx_train].reset_index(drop=True)
    df_holdout    = df_a.iloc[idx_holdout].reset_index(drop=True)
    df_holdout.to_csv(os.path.join(DATA_DIR, 'fixed_holdout.csv'), index=False)
    return df_train_pool, df_holdout, le

def run_mcnemar(df_train_a, df_train_b, df_holdout, le, random_state=RANDOM_STATE):
    pass  # dead code — lihat versi sebelumnya untuk implementasi lengkap


# ==============================================================
# PREPROCESSING
# ==============================================================

def preprocess(df, scaler=None, le=None, fit=True):
    X = df[FEATURE_COLS].fillna(0).values
    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    if le is None:
        le = LabelEncoder()
        y = le.fit_transform(df[LABEL_COL])
    else:
        y = le.transform(df[LABEL_COL])
    return X, y, scaler, le


# ==============================================================
# PAIRED T-TEST — dijalankan per model
# ==============================================================

def run_paired_ttest_single(model, model_name, X_a, y_a, X_b, y_b):
    """Jalankan paired t-test untuk satu model."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=False)

    scores_a = cross_val_score(model, X_a, y_a, cv=cv, scoring='f1_weighted', n_jobs=N_JOBS)

    # Clone model agar tidak ada state carry-over antar dataset
    from sklearn.base import clone
    model_b = clone(model)
    scores_b = cross_val_score(model_b, X_b, y_b, cv=cv, scoring='f1_weighted', n_jobs=N_JOBS)

    d       = scores_b - scores_a
    d_bar   = float(d.mean())
    s_d     = float(d.std(ddof=1))
    t_stat, p_value = stats.ttest_rel(scores_b, scores_a)

    logger.info(f"\n  [{model_name}]")
    logger.info(f"  F1 per fold A : {np.round(scores_a, 4)}")
    logger.info(f"  F1 per fold B : {np.round(scores_b, 4)}")
    logger.info(f"  Mean±Std  A   : {scores_a.mean():.4f} ± {scores_a.std():.4f}")
    logger.info(f"  Mean±Std  B   : {scores_b.mean():.4f} ± {scores_b.std():.4f}")
    logger.info(f"  d̄={d_bar:.4f}, s_d={s_d:.4f}, t({N_SPLITS-1})={t_stat:.4f}, p={p_value:.4f}")
    logger.info(f"  → {'H₀ DITOLAK' if p_value < 0.05 else 'H₀ GAGAL DITOLAK'}")

    return {
        'model'      : model_name,
        'scores_a'   : [round(float(s), 4) for s in scores_a],
        'scores_b'   : [round(float(s), 4) for s in scores_b],
        'mean_a'     : round(float(scores_a.mean()), 4),
        'std_a'      : round(float(scores_a.std()),  4),
        'mean_b'     : round(float(scores_b.mean()), 4),
        'std_b'      : round(float(scores_b.std()),  4),
        'd_bar'      : round(d_bar,          4),
        's_d'        : round(s_d,            4),
        't_stat'     : round(float(t_stat),  4),
        'p_value'    : round(float(p_value), 4),
        'df'         : N_SPLITS - 1,
        'significant': bool(p_value < 0.05)
    }


def run_paired_ttest_all_models(df_a, df_b, le):
    """Jalankan paired t-test untuk semua 3 model."""
    logger.info(f"\n=== PAIRED T-TEST ({N_SPLITS}-Fold CV) — 3 MODEL ===")

    X_a, y_a, _, _ = preprocess(df_a, le=le, fit=True)
    X_b, y_b, _, _ = preprocess(df_b, le=le, fit=True)

    results = {}
    for model_name, model in MODELS.items():
        results[model_name] = run_paired_ttest_single(model, model_name, X_a, y_a, X_b, y_b)

    return results


# ==============================================================
# EVALUASI DESKRIPTIF — 3 model sekaligus
# ==============================================================

def evaluate_descriptive_all(df, dataset_name, le):
    """Evaluasi deskriptif F1 weighted untuk semua 3 model."""
    X, y, _, _ = preprocess(df, le=le, fit=True)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=False)

    results = {'dataset': dataset_name, 'n_samples': len(df)}
    for model_name, model in MODELS.items():
        from sklearn.base import clone
        m = clone(model)
        scores = cross_val_score(m, X, y, cv=cv, scoring='f1_weighted', n_jobs=N_JOBS)
        results[model_name] = {
            'f1_mean': round(float(scores.mean()), 4),
            'f1_std' : round(float(scores.std()),  4)
        }
        logger.info(f"  {dataset_name} | {model_name}: {scores.mean():.4f} ± {scores.std():.4f}")

    return results


# ==============================================================
# MAIN
# ==============================================================

def main():
    logger.info("=" * 60)
    logger.info("EVALUATION — 3 Model: RF, DT, SVM + Paired T-Test")
    logger.info("=" * 60)

    df_a = pd.read_csv(DATASET_A)
    df_b = pd.read_csv(DATASET_B)
    logger.info(f"Dataset A (AHP) : {len(df_a)} sampel")
    logger.info(f"Dataset B (GT)  : {len(df_b)} sampel")

    le = LabelEncoder()
    le.fit(df_a[LABEL_COL])
    logger.info(f"Kelas label     : {list(le.classes_)}")

    # Evaluasi deskriptif
    logger.info("\n=== EVALUASI DESKRIPTIF ===")
    desc_a = evaluate_descriptive_all(df_a, "Dataset A (AHP)", le)
    desc_b = evaluate_descriptive_all(df_b, "Dataset B (GT)",  le)

    # Paired t-test per model
    ttest_results = run_paired_ttest_all_models(df_a, df_b, le)

    # Simpan output
    output = {
        'descriptive_a': desc_a,
        'descriptive_b': desc_b,
        'paired_ttest' : ttest_results
    }
    out_file = os.path.join(OUTPUT_DIR, 'statistical_validation_results.json')
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)

    # Ringkasan untuk tabel tesis
    logger.info("\n" + "=" * 60)
    logger.info("RINGKASAN UNTUK TABEL TESIS")
    logger.info("=" * 60)
    logger.info(f"{'Model':<18} {'F1 A (mean±std)':<22} {'F1 B (mean±std)':<22} {'t':>8} {'p':>8} {'Sig?':>6}")
    logger.info("-" * 90)
    for model_name, res in ttest_results.items():
        sig = "YA" if res['significant'] else "TIDAK"
        logger.info(
            f"{model_name:<18} "
            f"{res['mean_a']:.4f} ± {res['std_a']:.4f}       "
            f"{res['mean_b']:.4f} ± {res['std_b']:.4f}       "
            f"{res['t_stat']:>8.4f} "
            f"{res['p_value']:>8.4f} "
            f"{sig:>6}"
        )
    logger.info(f"\nHasil lengkap disimpan ke: {out_file}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)