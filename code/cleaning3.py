#!/usr/bin/env python3
"""
fair_pipeline.py - Fair Confident Learning + iFlipper Implementation (REVISED)

Revisi utama (menyesuaikan dengan draf tesis Bab 3.2.3):
1. Confident Learning: clf_A/clf_B sekarang dilatih ulang pada subset CLEAN
   yang diperbarui tiap epoch (iteratif progresif), bukan data grup penuh.
2. Assignment label estimasi (z_hat) TIDAK LAGI menggunakan prioritas
   "cek model B dulu, fallback ke A" (tidak ada dasarnya di draf tesis).
   Sekarang setiap sampel dievaluasi oleh model sesuai kelompok asalnya:
   sampel privileged -> model B, sampel disadvantaged -> model A.
   CATATAN: Ini pilihan desain terbaik yang dipilih karena draf tesis 3.2.3
   tidak menjelaskan skema penentuan model mana yang dipakai. PERLU
   DITAMBAHKAN penjelasan skema ini ke naskah tesis.
3. Confident Joint (C_bar) & Prune by Class (PBC) DIHAPUS dari pipeline.
   Kedua konsep ini hanya disebut di tinjauan pustaka (Bab 2) sebagai bagian
   dari algoritma ORISINAL Zhang et al., namun TIDAK dideskripsikan sebagai
   bagian dari algoritma yang diimplementasikan di Bab 3.2.3. Deteksi noise
   sekarang murni menggunakan kriteria inkonsistensi label (y_n != z_hat_n)
   sesuai naskah tesis.
4. iFlipper: Adaptasi multi-kelas menggunakan pendekatan greedy per-sampel
   dengan voting berbasis reduction score, menghindari overwrite bergantian.
5. Perhitungan metrik fairness (SPD/DI/EOD/IF) DIHAPUS dari file ini sesuai
   permintaan; pipeline ini sekarang hanya mencakup tahap Confident Learning
   + iFlipper (tanpa evaluasi metrik keadilan).

Berdasarkan:
- Northcutt et al. (2021): "Confident Learning: Estimating Uncertainty in Dataset Labels"
- Zhang et al. (2023): "iFlipper: Label Flipping for Individual Fairness"
"""
import csv, os, sys, json, logging, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linprog
import traceback

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASET_A = os.path.join(DATA_DIR, 'final_sdn_dataset_ahp.csv')
DATASET_A_CLEAN = os.path.join(DATA_DIR, 'DATASET_CLEAN.csv')
DATASET_B = os.path.join(DATA_DIR, 'GROUND_TRUTH.csv')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

FAIRNESS_CONFIG = {
    'privileged_groups': ['linear', 'star', 'tree'],
    'disadvantaged_groups': ['mesh', 'ring'],
    'sensitive_attr': 'topology_type'
}


def load_dataset(filepath):
    if not os.path.exists(filepath):
        logger.error(f"Dataset not found: {filepath}")
        return None
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} samples from {os.path.basename(filepath)}")
    return df


# ============================================================================
# CONFIDENT LEARNING IMPLEMENTATION (REVISED - Iterative Progressive)
# ============================================================================

def psi(x):
    """Eq. (34): Confidence transformation function"""
    return np.log(1 + x + (x ** 2) / 2)


def compute_threshold(probs, labels, n_classes, Ns, nu):
    """
    Eq. (35)-(36): Compute dynamic threshold µ_j per class
    Uses lower confidence bound based on transformed probabilities.
    """
    thresholds = {}
    for j in range(n_classes):
        idx = np.where(labels == j)[0]
        if len(idx) == 0:
            thresholds[j] = 0.0
            continue

        p_j = probs[idx, j]
        t_bar = np.mean(psi(p_j))

        Q = nu * (Ns + (nu * np.log(2 * Ns)) / (Ns ** 2))
        mu_j = t_bar - Q / (Ns - nu)

        thresholds[j] = max(mu_j, 0.0)  # Ensure non-negative
    return thresholds


def assign_true_labels(probs_A, probs_B, labels, thresh_A, thresh_B, n_classes, s):
    """
    REVISI: Assign estimated true labels berdasarkan kelompok asal sampel
    (bukan prioritas cek model B lalu fallback ke A).

    CATATAN DESAIN (BELUM ADA DI DRAF TESIS 3.2.3 - PERLU DITAMBAHKAN):
    Draf tesis tidak menjelaskan model mana yang dipakai/diprioritaskan saat
    menentukan z_hat untuk satu sampel. Karena clf_A dan clf_B masing-masing
    dilatih khusus untuk kelompok Disadvantaged dan Privileged, pilihan
    desain yang paling konsisten dengan tujuan pemisahan model tersebut
    (agar noise mencerminkan bias topologi spesifik kelompok, bukan
    ketidakpastian klasifikasi lintas kelompok) adalah:
      - Sampel dari kelompok Privileged (s=1) dievaluasi memakai model B.
      - Sampel dari kelompok Disadvantaged (s=0) dievaluasi memakai model A.
    Tidak ada mekanisme fallback ke model lain.

    Parameters:
        s: array kelompok sensitif (1 = privileged, 0 = disadvantaged)
    """
    N = len(labels)
    z_hat = np.full(N, -1, dtype=int)
    source = np.full(N, '', dtype=object)

    for n in range(N):
        y_n = labels[n]

        if s[n] == 1:
            # Sampel privileged -> gunakan model B
            p = probs_B[n, y_n]
            if p >= thresh_B.get(y_n, 0):
                z_hat[n] = int(np.argmax(probs_B[n]))
                source[n] = 'B'
            else:
                source[n] = 'none'
        else:
            # Sampel disadvantaged -> gunakan model A
            p = probs_A[n, y_n]
            if p >= thresh_A.get(y_n, 0):
                z_hat[n] = int(np.argmax(probs_A[n]))
                source[n] = 'A'
            else:
                source[n] = 'none'

    return z_hat, source


def fair_confident_learning(df, epochs=50, Ns_fraction=0.6, nu=1e-2):
    """
    Algorithm 1 (REVISED): Fairness through Confident Learning - Iterative Progressive

    REVISI UTAMA:
    - clf_A dan clf_B sekarang dilatih ulang pada subset CLEAN yang diperbarui
      tiap epoch, bukan data grup penuh yang sama.
    - Confident joint C_bar digunakan untuk Prune by Class (PBC) sebagai
      mekanisme utama identifikasi noise, bukan hanya off-diagonal check.
    - Iterasi benar-benar progresif: subset clean menyempit, model menjadi
      lebih akurat dalam mendeteksi noise tersisa.

    Parameters:
        df: input dataframe
        epochs: number of training epochs
        Ns_fraction: fraction of estimated clean instances
        nu: variance parameter
    """
    logger.info("=" * 60)
    logger.info("FAIR CONFIDENT LEARNING (REVISED - Iterative Progressive)")
    logger.info("=" * 60)

    feature_cols = ['throughput_mbps', 'latency_ms', 'packet_loss_percent', 
                  'jitter_ms', 'cpu_usage_percent', 'memory_usage_mb',
                  'flow_setup_time_ms', 'nodes']

    df_copy = df.copy()
    df_features = df_copy[feature_cols].fillna(0)

    scaler = StandardScaler()
    X = scaler.fit_transform(df_features)

    le = LabelEncoder()
    y = le.fit_transform(df_copy['best_controller_label'])
    n_classes = len(le.classes_)

    s = df_copy[FAIRNESS_CONFIG['sensitive_attr']].isin(
        FAIRNESS_CONFIG['privileged_groups']
    ).astype(int).values

    N = len(y)
    Ns = max(int(Ns_fraction * N), 2)

    idx_A = np.where(s == 0)[0]  # disadvantaged
    idx_B = np.where(s == 1)[0]  # privileged

    logger.info(f"Total samples: {N}")
    logger.info(f"Privileged (B): {len(idx_B)} | Disadvantaged (A): {len(idx_A)}")
    logger.info(f"Estimated clean: {Ns}")

    # Inisialisasi: semua sampel dianggap clean di awal
    clean_mask = np.ones(N, dtype=bool)
    all_biased = set()

    for epoch in range(epochs):
        # REVISI: Latih model HANYA pada subset clean yang diperbarui
        current_clean = np.where(clean_mask)[0]

        # Pisahkan clean indices per grup
        clean_A = np.intersect1d(current_clean, idx_A)
        clean_B = np.intersect1d(current_clean, idx_B)

        # Inisialisasi model baru tiap epoch (tidak carry-over parameter)
        clf_A = LogisticRegression(max_iter=1000, random_state=42)
        clf_B = LogisticRegression(max_iter=1000, random_state=42)

        # Latih pada subset clean SAJA (bukan data grup penuh)
        # REVISI: cek jumlah SAMPEL sekaligus jumlah KELAS UNIK di subset clean.
        # Subset clean bisa cukup besar tapi kehabisan salah satu kelas label
        # sepenuhnya, yang membuat LogisticRegression gagal (butuh >=2 kelas).
        if len(clean_A) >= n_classes and len(np.unique(y[clean_A])) >= n_classes:
            clf_A.fit(X[clean_A], y[clean_A])
        else:
            # Fallback: gunakan semua data grup jika clean subset terlalu kecil/tidak lengkap kelasnya
            clf_A.fit(X[idx_A], y[idx_A])

        if len(clean_B) >= n_classes and len(np.unique(y[clean_B])) >= n_classes:
            clf_B.fit(X[clean_B], y[clean_B])
        else:
            clf_B.fit(X[idx_B], y[idx_B])

        # Prediksi probabilitas untuk SEMUA sampel
        all_classes = np.arange(n_classes)

        def predict_proba_full(clf, X_input, all_classes):
            raw = clf.predict_proba(X_input)
            full = np.zeros((len(X_input), len(all_classes)))
            for i, c in enumerate(clf.classes_):
                full[:, c] = raw[:, i]
            return full

        probs_A = predict_proba_full(clf_A, X, all_classes)
        probs_B = predict_proba_full(clf_B, X, all_classes)

        # Hitung threshold dinamis
        thresh_A = compute_threshold(probs_A, y, n_classes, Ns, nu)
        thresh_B = compute_threshold(probs_B, y, n_classes, Ns, nu)

        # Assign estimated true labels (berdasarkan kelompok asal sampel, s)
        z_hat, source = assign_true_labels(probs_A, probs_B, y, thresh_A, thresh_B, n_classes, s)

        # Identifikasi noisy: sampel diterima namun y_n != z_hat_n (inkonsistensi label)
        # sesuai Bab 3.2.3 draf tesis. PBC/confident joint dihapus karena tidak
        # dideskripsikan sebagai bagian dari algoritma implementasi (hanya di Bab 2).
        off_diag_noisy = set()
        for n in range(N):
            if z_hat[n] != -1 and y[n] != z_hat[n]:
                off_diag_noisy.add(n)

        epoch_noisy = off_diag_noisy

        # REVISI: hitung sampel BARU (belum pernah ditandai noisy sebelumnya)
        # sebelum all_biased diperbarui, karena epoch_noisy bisa berisi sampel
        # lama yang terus terdeteksi ulang (sudah stabil tapi tidak "baru").
        new_noisy = epoch_noisy - all_biased
        all_biased.update(epoch_noisy)

        # REVISI: Update clean_mask - hanya sampel yang belum terdeteksi noisy
        clean_mask = np.ones(N, dtype=bool)
        clean_mask[list(all_biased)] = False

        if (epoch + 1) % 10 == 0 or epoch < 5:
            logger.info(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"New noisy: {len(new_noisy):4d} | "
                  f"Total noisy: {len(all_biased):4d} | "
                  f"Clean remaining: {np.sum(clean_mask):4d}")

        # Cek konvergensi: jika tidak ada sampel BARU yang terdeteksi noisy
        if len(new_noisy) == 0:
            logger.info(f"Converged at epoch {epoch+1}")
            break

    df_copy['is_noisy_cl'] = False
    df_copy.loc[list(all_biased), 'is_noisy_cl'] = True

    noisy_count = len(all_biased)
    logger.info(f"\nConfident Learning detected {noisy_count} noisy samples ({noisy_count/N*100:.1f}%)")
    logger.info(f"Clean samples remaining: {N - noisy_count}")

    return df_copy


# ============================================================================
# IFLIPPER IMPLEMENTATION (REVISED - Multi-Class Greedy)
# ============================================================================

def generate_similarity_matrix(X, method='knn', k=5):
    """
    Generate similarity matrix using KNN.
    Based on iFlipper: similar instances should be treated similarly.
    """
    n = len(X)

    if method == 'knn':
        nn = NearestNeighbors(n_neighbors=min(k+1, n), algorithm='ball_tree')
        nn.fit(X)
        _, indices = nn.kneighbors(X)

        W = np.zeros((n, n))
        for i in range(n):
            for j_idx in range(1, min(k+1, n)):
                j = indices[i, j_idx]
                W[i, j] = 1.0

        # Symmetrize
        W = np.maximum(W, W.T)
        np.fill_diagonal(W, 0)

    return W


def compute_weighted_mismatches(y, W):
    """
    Hitung total weighted mismatches (pelanggaran individual fairness).
    """
    n = len(y)
    total_error = 0.0
    edge_count = 0

    for i in range(n):
        for j in range(i + 1, n):
            if W[i, j] > 0:
                edge_count += 1
                if y[i] != y[j]:
                    total_error += W[i, j]

    return total_error, edge_count


def compute_flip_impact(y, W, idx, n_classes):
    """
    Hitung reduction dalam weighted mismatches jika label idx diubah ke setiap kelas.

    REVISI: Untuk multi-kelas, evaluasi dampak perubahan ke SEMUA kelas lain,
    bukan hanya biner 0/1.

    Returns: dict {new_label: reduction_score}
    """
    n = len(y)
    current_label = y[idx]
    impacts = {}

    for new_label in range(n_classes):
        if new_label == current_label:
            continue

        # Hitung error setelah flip
        new_error = 0.0
        for j in range(n):
            if W[idx, j] > 0:
                if new_label != y[j]:
                    new_error += W[idx, j]

        # Hitung error sebelum flip (hanya edge ke idx)
        old_error = 0.0
        for j in range(n):
            if W[idx, j] > 0:
                if current_label != y[j]:
                    old_error += W[idx, j]

        # Reduction = pengurangan error lokal
        # (Tidak menghitung ulang seluruh graph untuk efisiensi)
        reduction = old_error - new_error
        impacts[new_label] = reduction

    return impacts


def iflipper_multiclass(y, W, error_budget, n_classes):
    """
    REVISI: iFlipper untuk klasifikasi multi-kelas.

    Pendekatan: Greedy per-sampel dengan evaluasi dampak ke semua kelas.
    - Hitung weighted mismatches awal
    - Iteratif: pilih sampel dengan reduction terbesar, flip ke kelas terbaik
    - Hentikan jika error <= budget atau tidak ada reduction positif

    Justifikasi adaptasi multi-kelas:
    Paper iFlipper asli (Zhang et al., 2023) dirancang untuk klasifikasi biner.
    Adaptasi ke multi-kelas menggunakan pendekatan greedy yang mempertimbangkan
    semua kemungkinan kelas target untuk setiap flip, memilih yang memberikan
    pengurangan weighted mismatches terbesar. Ini konsisten dengan prinsip
    iFlipper: minimal label flips untuk memaksimalkan individual fairness.
    """
    n = len(y)
    y_flipped = y.copy()

    # Hitung error awal
    initial_error, edge_count = compute_weighted_mismatches(y_flipped, W)
    error_budget_val = initial_error * error_budget

    logger.info(f"iFlipper Multi-Class: Initial error={initial_error:.2f}, "
                f"edges={edge_count}, budget={error_budget_val:.2f} ({error_budget*100:.0f}%)")

    if initial_error <= error_budget_val:
        logger.info("Error already within budget, no flips needed.")
        return y_flipped, 0

    flips_done = 0
    max_iterations = n * n_classes  # Safety limit

    for iteration in range(max_iterations):
        current_error, _ = compute_weighted_mismatches(y_flipped, W)

        if current_error <= error_budget_val:
            break

        # Cari sampel dengan reduction terbaik
        best_reduction = 0
        best_idx = -1
        best_new_label = -1

        for idx in range(n):
            impacts = compute_flip_impact(y_flipped, W, idx, n_classes)

            for new_label, reduction in impacts.items():
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_idx = idx
                    best_new_label = new_label

        # Jika tidak ada reduction positif, hentikan
        if best_reduction <= 0 or best_idx == -1:
            logger.info(f"No positive reduction found at iteration {iteration+1}")
            break

        # Lakukan flip
        old_label = y_flipped[best_idx]
        y_flipped[best_idx] = best_new_label
        flips_done += 1

        if (flips_done % 50 == 0) or (iteration < 10):
            new_error, _ = compute_weighted_mismatches(y_flipped, W)
            logger.info(f"  Flip {flips_done}: idx={best_idx} {old_label}->{best_new_label} "
                      f"(reduction={best_reduction:.3f}), error={new_error:.2f}")

    final_error, _ = compute_weighted_mismatches(y_flipped, W)
    logger.info(f"iFlipper: {flips_done} labels flipped, error {initial_error:.2f}->{final_error:.2f}")

    return y_flipped, flips_done


def apply_iflipper_fairness(df, error_fraction=0.1):
    """
    Apply iFlipper algorithm for individual fairness (REVISED multi-class).

    REVISI: Tidak lagi menggunakan dekomposisi biner one-vs-rest yang
    menyebabkan overwrite bergantian. Sekarang menggunakan iflipper_multiclass
    yang bekerja langsung pada label multi-kelas.
    """
    logger.info("=" * 60)
    logger.info("IFLIPPER: Individual Fairness (REVISED - Multi-Class Greedy)")
    logger.info("=" * 60)

    feature_cols = ['throughput_mbps', 'latency_ms', 'packet_loss_percent', 
                   'jitter_ms', 'cpu_usage_percent', 'memory_usage_mb',
                   'flow_setup_time_ms', 'nodes']

    df_fair = df.copy()
    df_features = df_fair[feature_cols].fillna(0)

    scaler = StandardScaler()
    X = scaler.fit_transform(df_features)

    le = LabelEncoder()
    le.fit(df_fair['best_controller_label'].unique())
    y_original = le.transform(df_fair['best_controller_label'])
    n_classes = len(le.classes_)

    logger.info(f"Classes: {le.classes_} (n={n_classes})")
    logger.info(f"Generating similarity matrix for {len(X)} samples...")
    W = generate_similarity_matrix(X, method='knn', k=5)

    # Jalankan iFlipper multi-kelas
    y_flipped, total_flips = iflipper_multiclass(y_original, W, error_fraction, n_classes)

    # Update labels
    df_fair['best_controller_label'] = le.inverse_transform(y_flipped)

    # Track flips
    df_fair['iflipper_applied'] = (y_original != y_flipped)

    logger.info(f"iFlipper: {total_flips} labels modified for fairness")

    return df_fair


# ============================================================================
# CATATAN: Fungsi metrik fairness (SPD/DI/EOD/Individual Fairness) dan
# generate_fairness_report() dihapus dari file ini sesuai permintaan.
# Pipeline ini sekarang hanya mencakup Confident Learning + iFlipper.
# Evaluasi metrik keadilan dilakukan di skrip/tahap terpisah.
# ============================================================================
# MAIN PIPELINE
# ============================================================================

def apply_confident_learning_filter(df):
    """Apply Confident Learning noise detection (REVISED)."""
    df_cl = fair_confident_learning(df, epochs=50, Ns_fraction=0.6, nu=1e-2)
    return df_cl


def apply_iflipper_correction(df):
    """Apply iFlipper fairness correction (REVISED multi-class)."""
    return apply_iflipper_fairness(df, error_fraction=0.1)


def main():
    logger.info("=" * 60)
    logger.info("PIPELINE: CONFIDENT LEARNING + IFLIPPER (REVISED)")
    logger.info("=" * 60)

    df = load_dataset(DATASET_A)
    if df is None:
        logger.error("Please ensure dataset exists at: " + DATASET_A)
        return

    logger.info(f"Original dataset: {len(df)} samples")

    logger.info("\n" + "=" * 60)
    logger.info("Stage 1: CONFIDENT LEARNING (REVISED)")
    logger.info("=" * 60)
    df_cl = apply_confident_learning_filter(df)

    df_clean = df_cl[~df_cl.get('is_noisy_cl', pd.Series([False]*len(df_cl)))].copy()
    if 'is_noisy_cl' in df_clean.columns:
        df_clean = df_clean.drop(columns=['is_noisy_cl'])

    noisy_removed = len(df) - len(df_clean)
    logger.info(f"Removed {noisy_removed} noisy samples, kept {len(df_clean)}")

    df_clean.to_csv(DATASET_A_CLEAN, index=False)
    logger.info(f"Clean dataset saved: {DATASET_A_CLEAN}")

    logger.info("\n" + "=" * 60)
    logger.info("Stage 2: IFLIPPER (REVISED Multi-Class)")
    logger.info("=" * 60)
    df_fair = apply_iflipper_correction(df_clean)

    df_fair.to_csv(DATASET_B, index=False)
    logger.info(f"Fair dataset saved: {DATASET_B}")

    pipeline_results = {
        'original_samples': len(df),
        'clean_samples': len(df_clean),
        'fair_samples': len(df_fair),
        'noise_removed': noisy_removed,
    }

    results_file = os.path.join(OUTPUT_DIR, 'pipeline_results.json')
    with open(results_file, 'w') as f:
        json.dump(pipeline_results, f, indent=2)
    logger.info(f"\nResults saved: {results_file}")

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Original (AHP): {pipeline_results['original_samples']}")
    logger.info(f"After CL:       {pipeline_results['clean_samples']}")
    logger.info(f"After iFlipper: {pipeline_results['fair_samples']}")

    return pipeline_results


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
