#!/usr/bin/env python3
"""
advanced_statistics.py

Implementasi analisis statistik lanjutan sesuai draf tesis Bab 3.2.5
(paragraf "Analisis deskriptif", "Analisis inferensial", dan
"Penguatan validitas"):
  1. Statistik deskriptif (mean/median/std/IQR) per metrik teknis
  2. Distribusi label kontroler per topologi
  3. Boxplot per kombinasi topologi-kontroler (per metrik teknis)
  4. Paired t-test (alpha=0.05): akurasi Dataset A vs Ground Truth
  5. Chi-square test: asosiasi label vs topologi
  6. SHAP Values: dominasi fitur topologi dalam keputusan model
  7. Noise Transition Matrix: pola perubahan label Dataset -> Ground Truth
  8. Bootstrap Confidence Interval (1000 iterasi): RF metrics & fairness metrics

CATATAN ASUMSI YANG PERLU DITAMBAHKAN/DIKONFIRMASI KE NASKAH TESIS
(karena tidak dijelaskan secara eksplisit di metodologi):

[A] PAIRED T-TEST: metodologi tidak menjelaskan bagaimana mendapatkan
    beberapa nilai akurasi berpasangan (paired t-test butuh >1 pasangan,
    bukan cuma 1 angka akurasi vs 1 angka akurasi dari train_test_split
    tunggal). Solusi yang dipakai di sini: Repeated Stratified K-Fold
    (5-fold x 10 repeat = 50 pasangan skor akurasi) dijalankan TERPISAH
    pada Dataset A dan Ground Truth (RF dengan konfigurasi Tabel 3.7),
    lalu skor akurasi dipasangkan per indeks repeat/fold yang sama.
    Ini adalah praktik umum di literatur ML untuk paired t-test model,
    namun bukan "true pairing" (unit yang benar-benar sama), karena kedua
    dataset punya jumlah sampel berbeda (1050 vs 833). Perlu dicantumkan
    di Bab 3 sebagai definisi operasional "paired" yang dipakai.

[B] SHAP: metodologi menyebut "dominasi fitur topologi" tapi FEATURE_COLS
    RF utama (Tabel 3.7) tidak menyertakan topology_type (kategorikal).
    Karena itu, SHAP dijalankan pada MODEL DIAGNOSTIK TERPISAH (bukan
    model evaluasi utama) yang menyertakan topology_type (one-hot) DAN
    nodes (proxy kompleksitas topologi kontinu) sebagai fitur tambahan.
    "Dominasi fitur topologi" dihitung sebagai proporsi total |SHAP value|
    dari fitur topology_type (one-hot) + nodes, dibandingkan Dataset A vs
    Ground Truth. Ini perlu dijelaskan di Bab 3 sebagai model terpisah
    dari RF evaluasi utama.

[C] BOOTSTRAP CI: diterapkan ke DUA kelompok metrik: RF (accuracy,
    precision, recall, f1) dan fairness (SPD, DI, EOD, IF), masing-masing
    dari 1000 iterasi bootstrap resampling pada TEST SET (bukan retrain
    ulang model 1000x, karena terlalu mahal secara komputasi) -- model RF
    dilatih SEKALI, lalu prediksinya di-bootstrap pada level baris test
    set untuk mengestimasi CI. Ini pendekatan non-parametric bootstrap CI
    standar. Perlu dicantumkan di Bab 3 bahwa CI dihitung dari resampling
    prediksi test set, bukan dari pelatihan ulang model per iterasi.

[D] NOISE TRANSITION MATRIX: dihitung dengan mencocokkan baris via kolom
    'sample_id' yang tersedia di ketiga dataset (final_sdn_dataset_ahp.csv,
    DATASET_CLEAN.csv, GROUND_TRUTH.csv). Sampel yang dibuang saat
    Confident Learning (tidak ada di Ground Truth) TIDAK dimasukkan ke
    matrix ini (karena tidak punya "label baru" untuk dibandingkan) dan
    dilaporkan terpisah sebagai jumlah sampel yang dibuang.

[E] BOXPLOT: dibuat satu figur per metrik teknis, dengan boxplot
    dikelompokkan berdasarkan kombinasi topology_type x controller_name,
    dihitung pada Dataset A (kondisi "sebelum pipeline", sesuai kalimat
    metodologi "karakteristik performa awal sebelum pipeline").
"""
import os
import json
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import shap

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'results')
PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

DATASET_A = os.path.join(DATA_DIR, 'final_sdn_dataset_ahp.csv')
DATASET_A_CLEAN = os.path.join(DATA_DIR, 'DATASET_CLEAN.csv')
DATASET_B = os.path.join(DATA_DIR, 'GROUND_TRUTH.csv')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TECHNICAL_METRICS = [
    'throughput_mbps', 'latency_ms', 'packet_loss_percent', 'jitter_ms',
    'flow_setup_time_ms', 'cpu_usage_percent', 'memory_usage_mb'
]

FEATURE_COLS = TECHNICAL_METRICS + ['nodes']

FAIRNESS_CONFIG = {
    'privileged_groups': ['linear', 'star', 'tree'],
    'disadvantaged_groups': ['mesh', 'ring'],
    'sensitive_attr': 'topology_type'
}

LABEL_COL = 'best_controller_label'
SENSITIVE_COL = 'topology_type'
ID_COL = 'sample_id'

RANDOM_STATE = 42
N_BOOTSTRAP = 1000
ALPHA = 0.05


def load_dataset(filepath):
    if not os.path.exists(filepath):
        logger.error(f"Dataset not found: {filepath}")
        return None
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} samples from {os.path.basename(filepath)}")
    return df


# ============================================================================
# 1-3. ANALISIS DESKRIPTIF + DISTRIBUSI LABEL + BOXPLOT
# ============================================================================

def descriptive_statistics(df):
    """
    Mean, median, std, IQR (Q1, Q3, IQR) untuk tiap metrik teknis.
    """
    results = {}
    for col in TECHNICAL_METRICS:
        series = df[col].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        results[col] = {
            'mean': round(float(series.mean()), 4),
            'median': round(float(series.median()), 4),
            'std': round(float(series.std()), 4),
            'q1': round(float(q1), 4),
            'q3': round(float(q3), 4),
            'iqr': round(float(q3 - q1), 4),
        }
    return results


def label_distribution_per_topology(df):
    """
    Distribusi label kontroler per topologi (indikator awal bias sistematis).
    """
    crosstab = pd.crosstab(df[SENSITIVE_COL], df[LABEL_COL])
    crosstab_pct = pd.crosstab(df[SENSITIVE_COL], df[LABEL_COL], normalize='index') * 100
    return crosstab, crosstab_pct


def generate_boxplots(df, dataset_label, output_dir):
    """
    Boxplot per kombinasi topologi-kontroler, untuk tiap metrik teknis
    (mendeteksi sebaran, outlier, karakteristik performa awal).
    """
    df = df.copy()
    df['combo'] = df[SENSITIVE_COL] + " - " + df['controller_name']
    combos = sorted(df['combo'].unique())

    saved_files = []
    for metric in TECHNICAL_METRICS:
        fig, ax = plt.subplots(figsize=(12, 6))
        data_by_combo = [df[df['combo'] == c][metric].dropna().values for c in combos]
        ax.boxplot(data_by_combo, labels=combos, showfliers=True)
        ax.set_title(f"Boxplot {metric} per Topologi-Kontroler ({dataset_label})")
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()

        filename = os.path.join(output_dir, f"boxplot_{metric}_{dataset_label}.png")
        plt.savefig(filename, dpi=120)
        plt.close(fig)
        saved_files.append(filename)
        logger.info(f"Boxplot saved: {filename}")

    return saved_files


# ============================================================================
# 4-5. ANALISIS INFERENSIAL: PAIRED T-TEST & CHI-SQUARE
# ============================================================================

def get_features_labels(df):
    df_features = df[FEATURE_COLS].fillna(0)
    scaler = StandardScaler()
    X = scaler.fit_transform(df_features)
    le = LabelEncoder()
    y = le.fit_transform(df[LABEL_COL])
    return X, y


def repeated_cv_accuracies(df, n_splits=5, n_repeats=10):
    """
    [ASUMSI A] Menghasilkan sampel akurasi berpasangan via Repeated
    Stratified K-Fold, karena metodologi tidak menjelaskan sumber pasangan
    akurasi untuk paired t-test.
    """
    X, y = get_features_labels(df)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)

    accuracies = []
    for train_idx, test_idx in rskf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))

    return np.array(accuracies)


def paired_ttest_accuracy(df_a, df_b, n_splits=5, n_repeats=10):
    """
    Paired t-test (alpha=0.05) membandingkan akurasi Dataset A vs Ground Truth.
    """
    logger.info("Menjalankan Repeated Stratified K-Fold untuk Dataset A...")
    acc_a = repeated_cv_accuracies(df_a, n_splits, n_repeats)
    logger.info("Menjalankan Repeated Stratified K-Fold untuk Ground Truth...")
    acc_b = repeated_cv_accuracies(df_b, n_splits, n_repeats)

    t_stat, p_value = stats.ttest_rel(acc_b, acc_a)

    result = {
        'n_pairs': len(acc_a),
        'mean_accuracy_dataset_a': round(float(acc_a.mean()), 4),
        'std_accuracy_dataset_a': round(float(acc_a.std()), 4),
        'mean_accuracy_ground_truth': round(float(acc_b.mean()), 4),
        'std_accuracy_ground_truth': round(float(acc_b.std()), 4),
        't_statistic': round(float(t_stat), 4),
        'p_value': float(p_value),
        'alpha': ALPHA,
        'significant': bool(p_value < ALPHA),
        'interpretation': (
            "Perbedaan akurasi signifikan secara statistik (bias struktural terdeteksi)"
            if p_value < ALPHA else
            "Perbedaan akurasi tidak signifikan secara statistik"
        )
    }
    logger.info(f"Paired t-test: t={t_stat:.4f}, p={p_value:.6f}, signifikan={result['significant']}")
    return result


def chi_square_label_topology(df, dataset_label):
    """
    Chi-square test: asosiasi antara label kontroler dan topologi.
    """
    crosstab = pd.crosstab(df[SENSITIVE_COL], df[LABEL_COL])
    chi2, p_value, dof, expected = stats.chi2_contingency(crosstab)

    result = {
        'dataset': dataset_label,
        'chi2_statistic': round(float(chi2), 4),
        'degrees_of_freedom': int(dof),
        'p_value': float(p_value),
        'alpha': ALPHA,
        'significant_association': bool(p_value < ALPHA),
        'interpretation': (
            "Ada asosiasi signifikan antara label dan topologi (indikasi bias struktural)"
            if p_value < ALPHA else
            "Tidak ada asosiasi signifikan antara label dan topologi"
        )
    }
    logger.info(f"Chi-square ({dataset_label}): chi2={chi2:.4f}, p={p_value:.6f}, "
                f"asosiasi signifikan={result['significant_association']}")
    return result


# ============================================================================
# 6. SHAP VALUES: DOMINASI FITUR TOPOLOGI
# ============================================================================

def shap_topology_dominance(df, dataset_label):
    """
    [ASUMSI B] Model diagnostik terpisah dari RF evaluasi utama:
    menyertakan topology_type (one-hot) + nodes sebagai fitur eksplisit,
    untuk mengukur dominasi fitur topologi dalam keputusan model.
    """
    df_diag = df.copy()
    topo_dummies = pd.get_dummies(df_diag[SENSITIVE_COL], prefix='topo')
    topo_feature_names = list(topo_dummies.columns)

    X_df = pd.concat([
        df_diag[TECHNICAL_METRICS + ['nodes']].fillna(0).reset_index(drop=True),
        topo_dummies.reset_index(drop=True)
    ], axis=1)

    feature_names = list(X_df.columns)
    topo_related_features = topo_feature_names + ['nodes']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    le = LabelEncoder()
    y = le.fit_transform(df_diag[LABEL_COL])

    model = RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1
    )
    model.fit(X_scaled, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    # shap_values bisa berbentuk list (per kelas) atau array 3D tergantung versi shap
    if isinstance(shap_values, list):
        abs_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    elif shap_values.ndim == 3:
        abs_shap = np.abs(shap_values).mean(axis=2)
    else:
        abs_shap = np.abs(shap_values)

    mean_abs_shap_per_feature = abs_shap.mean(axis=0)

    importance = dict(zip(feature_names, [round(float(v), 6) for v in mean_abs_shap_per_feature]))
    total_importance = sum(importance.values())

    topo_importance = sum(importance[f] for f in topo_related_features)
    topo_dominance_ratio = topo_importance / total_importance if total_importance > 0 else 0

    # Simpan summary plot
    plt.figure()
    shap.summary_plot(
        abs_shap if not isinstance(shap_values, list) else shap_values,
        X_scaled, feature_names=feature_names, show=False, plot_type='bar'
    )
    plot_path = os.path.join(PLOT_DIR, f"shap_summary_{dataset_label}.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    plt.close()

    result = {
        'dataset': dataset_label,
        'feature_importance_mean_abs_shap': importance,
        'topology_related_features': topo_related_features,
        'topology_importance_sum': round(float(topo_importance), 6),
        'total_importance_sum': round(float(total_importance), 6),
        'topology_dominance_ratio': round(float(topo_dominance_ratio), 4),
        'plot_path': plot_path,
    }
    logger.info(f"SHAP ({dataset_label}): topology_dominance_ratio={topo_dominance_ratio:.4f}")
    return result


# ============================================================================
# 7. NOISE TRANSITION MATRIX
# ============================================================================

def noise_transition_matrix(df_a, df_b):
    """
    [ASUMSI D] Dicocokkan via sample_id. Sampel yang dibuang saat Confident
    Learning (tidak ada di Ground Truth) dilaporkan terpisah.
    """
    merged = df_a[[ID_COL, LABEL_COL]].merge(
        df_b[[ID_COL, LABEL_COL]], on=ID_COL, suffixes=('_dataset', '_groundtruth')
    )

    n_matched = len(merged)
    n_removed = len(df_a) - n_matched

    classes = sorted(set(df_a[LABEL_COL].unique()) | set(df_b[LABEL_COL].unique()))
    matrix = pd.crosstab(
        merged[f'{LABEL_COL}_dataset'], merged[f'{LABEL_COL}_groundtruth']
    ).reindex(index=classes, columns=classes, fill_value=0)

    n_changed = int((merged[f'{LABEL_COL}_dataset'] != merged[f'{LABEL_COL}_groundtruth']).sum())
    n_unchanged = n_matched - n_changed

    result = {
        'n_samples_dataset_a': len(df_a),
        'n_samples_matched_in_ground_truth': n_matched,
        'n_samples_removed_by_confident_learning': n_removed,
        'n_labels_changed_by_iflipper': n_changed,
        'n_labels_unchanged': n_unchanged,
        'transition_matrix': matrix.to_dict(),
    }
    logger.info(f"Noise Transition Matrix: {n_matched} sampel cocok, "
                f"{n_removed} dibuang CL, {n_changed} label diflip iFlipper")
    return result, matrix


# ============================================================================
# 8. BOOTSTRAP CONFIDENCE INTERVAL (1000 iterasi)
# ============================================================================

def get_favorable_outcome(labels_array):
    """Proxy favorable: label_encoded > 0 (konsisten dgn evaluation.py)."""
    return (labels_array > 0).astype(int)


def bootstrap_ci_metrics(df, dataset_label, n_bootstrap=N_BOOTSTRAP):
    """
    [ASUMSI C] Model dilatih SEKALI (train_test_split 80/20 sesuai Tabel 3.7),
    lalu prediksi test set di-bootstrap resample n_bootstrap kali untuk
    mengestimasi CI metrik RF dan fairness.
    """
    X, y = get_features_labels(df)
    indices = np.arange(len(df))

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices, test_size=0.20, random_state=RANDOM_STATE
    )
    df_test = df.iloc[idx_test].reset_index(drop=True)

    model = RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    n_test = len(y_test)
    rng = np.random.RandomState(RANDOM_STATE)

    metric_samples = {
        'accuracy': [], 'precision': [], 'recall': [], 'f1': [],
        'spd': [], 'di': [], 'eod': [], 'individual_fairness': []
    }

    privileged_mask_full = df_test[SENSITIVE_COL].isin(FAIRNESS_CONFIG['privileged_groups']).values
    disadvantaged_mask_full = df_test[SENSITIVE_COL].isin(FAIRNESS_CONFIG['disadvantaged_groups']).values
    y_true_favorable_full = get_favorable_outcome(y_test)
    y_pred_favorable_full = get_favorable_outcome(y_pred)

    for i in range(n_bootstrap):
        boot_idx = rng.choice(n_test, size=n_test, replace=True)

        yt, yp = y_test[boot_idx], y_pred[boot_idx]
        metric_samples['accuracy'].append(accuracy_score(yt, yp))
        metric_samples['precision'].append(precision_score(yt, yp, average='weighted', zero_division=0))
        metric_samples['recall'].append(recall_score(yt, yp, average='weighted', zero_division=0))
        metric_samples['f1'].append(f1_score(yt, yp, average='weighted', zero_division=0))

        priv_mask = privileged_mask_full[boot_idx]
        dis_mask = disadvantaged_mask_full[boot_idx]
        yt_fav = y_true_favorable_full[boot_idx]
        yp_fav = y_pred_favorable_full[boot_idx]

        p_priv = yp_fav[priv_mask].mean() if priv_mask.sum() > 0 else 0
        p_dis = yp_fav[dis_mask].mean() if dis_mask.sum() > 0 else 0
        spd = p_priv - p_dis
        di = p_dis / p_priv if p_priv > 0 else (0 if p_dis == 0 else np.inf)

        priv_true_pos = priv_mask & (yt_fav == 1)
        dis_true_pos = dis_mask & (yt_fav == 1)
        tpr_priv = yp_fav[priv_true_pos].mean() if priv_true_pos.sum() > 0 else 0
        tpr_dis = yp_fav[dis_true_pos].mean() if dis_true_pos.sum() > 0 else 0
        eod = tpr_priv - tpr_dis

        # Individual fairness disederhanakan: proporsi prediksi favorable sama
        # dengan proporsi keseluruhan (proxy cepat untuk 1000 iterasi; versi
        # lengkap KNN-based terlalu mahal dihitung 1000x pada tiap bootstrap)
        individual_fairness = 1.0 - np.abs(yp_fav.mean() - yt_fav.mean())

        metric_samples['spd'].append(spd)
        metric_samples['di'].append(di if np.isfinite(di) else np.nan)
        metric_samples['eod'].append(eod)
        metric_samples['individual_fairness'].append(individual_fairness)

    ci_results = {}
    for metric_name, samples in metric_samples.items():
        arr = np.array(samples, dtype=float)
        arr = arr[np.isfinite(arr)]
        lower = np.percentile(arr, 2.5)
        upper = np.percentile(arr, 97.5)
        ci_results[metric_name] = {
            'mean': round(float(arr.mean()), 4),
            'std': round(float(arr.std()), 4),
            'ci_lower_2.5%': round(float(lower), 4),
            'ci_upper_97.5%': round(float(upper), 4),
            'n_bootstrap': len(arr),
        }

    logger.info(f"Bootstrap CI selesai untuk {dataset_label} ({n_bootstrap} iterasi)")
    return {'dataset': dataset_label, 'bootstrap_ci': ci_results}


# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("=" * 60)
    logger.info("ANALISIS STATISTIK LANJUTAN")
    logger.info("=" * 60)

    df_a = load_dataset(DATASET_A)
    df_b = load_dataset(DATASET_B)

    if df_a is None or df_b is None:
        logger.error("Dataset tidak lengkap, proses dihentikan.")
        return

    all_results = {}

    # --- 1-2. Deskriptif + distribusi label ---
    logger.info("\n--- Analisis Deskriptif (Dataset A, sebelum pipeline) ---")
    desc_stats = descriptive_statistics(df_a)
    crosstab_count, crosstab_pct = label_distribution_per_topology(df_a)
    all_results['descriptive_statistics'] = desc_stats
    all_results['label_distribution_count'] = crosstab_count.to_dict()
    all_results['label_distribution_percent'] = crosstab_pct.round(2).to_dict()

    # --- 3. Boxplot ---
    logger.info("\n--- Boxplot per Topologi-Kontroler (Dataset A) ---")
    boxplot_files = generate_boxplots(df_a, "DatasetA", PLOT_DIR)
    all_results['boxplot_files'] = boxplot_files

    # --- 4. Paired t-test ---
    logger.info("\n--- Paired t-test: Akurasi Dataset A vs Ground Truth ---")
    ttest_result = paired_ttest_accuracy(df_a, df_b)
    all_results['paired_ttest'] = ttest_result

    # --- 5. Chi-square ---
    logger.info("\n--- Chi-square: Label vs Topologi ---")
    chi2_a = chi_square_label_topology(df_a, "Dataset A")
    chi2_b = chi_square_label_topology(df_b, "Ground Truth")
    all_results['chi_square'] = {'dataset_a': chi2_a, 'ground_truth': chi2_b}

    # --- 6. SHAP ---
    logger.info("\n--- SHAP: Dominasi Fitur Topologi ---")
    shap_a = shap_topology_dominance(df_a, "DatasetA")
    shap_b = shap_topology_dominance(df_b, "GroundTruth")
    all_results['shap_analysis'] = {'dataset_a': shap_a, 'ground_truth': shap_b}

    # --- 7. Noise Transition Matrix ---
    logger.info("\n--- Noise Transition Matrix ---")
    ntm_result, ntm_matrix = noise_transition_matrix(df_a, df_b)
    all_results['noise_transition_matrix'] = ntm_result
    ntm_matrix.to_csv(os.path.join(OUTPUT_DIR, 'noise_transition_matrix.csv'))

    # --- 8. Bootstrap CI ---
    logger.info("\n--- Bootstrap CI (1000 iterasi) ---")
    boot_a = bootstrap_ci_metrics(df_a, "Dataset A")
    boot_b = bootstrap_ci_metrics(df_b, "Ground Truth")
    all_results['bootstrap_ci'] = {'dataset_a': boot_a, 'ground_truth': boot_b}

    results_file = os.path.join(OUTPUT_DIR, 'advanced_statistics_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nSemua hasil disimpan di: {results_file}")

    logger.info("\n" + "=" * 60)
    logger.info("SELESAI")
    logger.info("=" * 60)

    return all_results


if __name__ == '__main__':
    main()
