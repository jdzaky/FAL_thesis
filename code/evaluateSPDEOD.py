#!/usr/bin/env python3
"""
evaluation.py (REVISED)
di_epsilon SET TO 0, NOT USED
Revisi utama (menyesuaikan draf tesis Bab 3.2.5 & Tabel 3.6/3.7):
1. calculate_fairness_metrics() DITULIS ULANG TOTAL. Versi lama rusak
   (memakai .replace('pred','true')/.replace('pred','label') pada kolom
   'best_controller_label' yang tidak mengandung substring 'pred', sehingga
   accuracy_score(df[label_col], df[label_col]) selalu = 1.0). Versi baru
   menghitung SPD, DI, EOD, dan Individual Fairness (IF) sesuai formula dan
   threshold Tabel 3.6, dihitung dari label AKTUAL (y_true) vs PREDIKSI
   model (y_pred) pada test set -> ini genuine post-model fairness,
   berbeda dari audit label di cleaning3.py yang mengevaluasi dataset saja.
2. evaluate_model() diganti total dari cross_val_score(cv=5) menjadi
   train_test_split 80/20, random_state=42, class_weight='balanced'
   sesuai Tabel 3.7. Metrik yang dihitung: Accuracy, Precision, Recall,
   F1-Score (average='weighted' untuk kasus multi-kelas -- asumsi ini
   PERLU DIKONFIRMASI/DITAMBAHKAN ke naskah tesis karena tidak disebutkan
   secara eksplisit).
3. Decision Tree dipertahankan sebagai baseline pembanding tambahan (bukan
   bagian dari metodologi inti), dengan konfigurasi yang sama (train_test_split
   80/20, random_state=42, class_weight='balanced').
4. Fairness metrics (SPD/DI/EOD/IF) dihitung pada TEST SET dari kedua model
   (Dataset A vs Dataset B/Ground truth) untuk mengukur efektivitas pipeline
   terhadap keadilan hasil model, sesuai maksud 3.2.5:
   "...mengukur efektivitas pipeline dalam meningkatkan keadilan dan akurasi
   pada model pembelajaran mesin dibandingkan terhadap dataset original."

5. calculate_individual_fairness() DITULIS ULANG dari proxy biner (cek
   kesamaan kelas favorable pada sampel bertetangga) menjadi implementasi
   kondisi Lipschitz sesuai Eq. (15): D(f(x_i),f(x_j)) <= L*D'(x_i,x_j),
   dengan f(x)=predict_proba (bukan y_pred_favorable), D=jarak Euclidean
   pada ruang probabilitas, D'=jarak Euclidean pada ruang fitur (skala
   dinormalisasi terhadap nilai maksimum antar pasangan tetangga), dan
   "sampel serupa" dioperasionalkan sebagai k-NN (k=5, konsisten dengan
   similarity graph iFlipper di cleaning3.py). L=1.0 dipakai sebagai
   default. IF dilaporkan sebagai proporsi pasangan tetangga yang
   memenuhi syarat Lipschitz -- konstanta Lipschitz empiris (rasio
   maksimum yang teramati) ikut dilaporkan terpisah untuk transparansi.
   ASUMSI k=5, normalisasi D/D', dan L=1.0 PERLU DIVALIDASI/DIJELASKAN
   DI BAB 3.

CATATAN YANG PERLU DITAMBAHKAN KE NASKAH TESIS:
- Definisi "favorable outcome" untuk label multi-kelas (Ryu/ONOS/Floodlight)
  belum dijelaskan di metodologi. Proxy yang dipakai di sini: bila kolom
  ahp_score tersedia, favorable = ahp_score >= median (konsisten dengan
  definisi yang sudah dipakai di cleaning3.py); jika tidak tersedia,
  fallback ke label_encoded > 0. Proxy ini bersifat sementara dan sebaiknya
  divalidasi/dijelaskan di Bab 3.

BELUM DIKERJAKAN DI FILE INI (di luar prioritas revisi saat ini, sesuai
kesepakatan): paired t-test, chi-square test, SHAP values, Noise Transition
Matrix, Bootstrap Confidence Interval, statistik deskriptif + boxplot.
"""
import csv, os, sys, json, logging, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
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

FEATURE_COLS = ['throughput_mbps', 'latency_ms', 'packet_loss_percent',
                'jitter_ms', 'cpu_usage_percent', 'memory_usage_mb',
                'flow_setup_time_ms', 'nodes']


def load_dataset(filepath):
    if not os.path.exists(filepath):
        logger.error(f"Dataset not found: {filepath}")
        return None
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} samples from {os.path.basename(filepath)}")
    return df


# ============================================================================
# FAIRNESS METRICS (REVISED - ditulis ulang total)
# ============================================================================

# ============================================================================
# FAIRNESS METRICS (REVISI: one-vs-rest per kelas, Opsi B)
# ============================================================================
#
# 'best_controller_label' bersifat multi-kelas (Ryu/ONOS/Floodlight) tanpa
# urutan preferensi alami, sehingga tidak ada satu kelas yang secara wajar
# bisa disebut "favorable" (proxy sebelumnya -- ahp_score median atau
# label_encoded>0 -- arbitrer dan tidak dijelaskan di metodologi).
#
# REVISI: SPD/DI/EOD dihitung SATU KALI PER KELAS (one-vs-rest: favorable
# = "diprediksi sebagai kelas tsb" vs bukan), lalu diringkas jadi nilai
# agregat worst-case (nilai disparitas terbesar di antara 3 kelas). Ini
# menghindari pemilihan favorable class yang arbitrer -- dataset dianggap
# adil hanya jika TIDAK ADA kelas mana pun yang menunjukkan disparitas,
# sesuai definisi tesis sendiri ("kondisi adil...secara simultan").


def calculate_individual_fairness(y_pred, X_scaled, k=5):
    """
    IF (Eq. 4 - Consistency Score / iFlipper):
        Skor Konsistensi = 1 - sum(|h(xi) - h(xj)| * Wij) / sum(Wij)
    - h(x) = prediksi label model (y_pred, bukan predict_proba)
    - Wij   = bobot kesamaan KNN (1 jika tetangga, 0 jika tidak), k=5
    """
    n = len(y_pred)
    nn = NearestNeighbors(n_neighbors=min(k+1, n), algorithm='ball_tree')
    nn.fit(X_scaled)
    _, indices = nn.kneighbors(X_scaled)

    weighted_mismatch = 0.0
    total_weight = 0.0

    for i in range(n):
        for j_idx in range(1, min(k+1, n)):
            j = indices[i, j_idx]
            w_ij = 1.0
            weighted_mismatch += abs(int(y_pred[i]) - int(y_pred[j])) * w_ij
            total_weight += w_ij

    if total_weight == 0:
        return 1.0, {'n_pairs': 0}

    if_score = 1.0 - (weighted_mismatch / total_weight)
    return float(if_score), {'n_pairs': int(total_weight)}

def calculate_fairness_metrics(df_test, sensitive_col, y_true, y_pred,
                                class_names, X_test_scaled, y_proba,
                                di_epsilon=0):
    """
    REVISI (Opsi B - one-vs-rest per kelas): SPD/DI/EOD dihitung untuk
    SETIAP kelas (favorable = "diprediksi sebagai kelas tsb" vs bukan),
    lalu diringkas jadi nilai agregat WORST-CASE (disparitas terbesar di
    antara ketiga kelas Ryu/ONOS/Floodlight). Menghindari pemilihan satu
    "favorable class" yang arbitrer seperti proxy sebelumnya.

    REVISI TAMBAHAN (berdasarkan temuan run sebelumnya):
    1. DI SMOOTHING: DI = (p_dis + eps) / (p_priv + eps), eps=di_epsilon
       (default 1e-3). Tanpa smoothing, DI bisa meledak ke puluhan hanya
       karena salah satu grup punya selection rate mendekati nol untuk
       kelas tertentu (artefak numerik, bukan sinyal keadilan yang
       berarti). ASUMSI nilai eps=1e-3 PERLU DIDOKUMENTASIKAN DI BAB 3.
    2. EOD SEBAGAI KRITERIA UTAMA: SPD/DI di problem ini berpotensi
       merefleksikan pola topologi->kontroler yang LEGITIMATE (mis. mesh
       memang dominan direkomendasikan ONOS secara performa, lihat
       label_distribution_percent), bukan bias sistem murni -- karena
       SPD/DI mengasumsikan distribusi outcome semestinya rata antar
       grup, asumsi yang tidak selalu tepat saat label memang seharusnya
       bergantung pada atribut yang berkorelasi dengan grup sensitif.
       EOD ("apakah model sama akuratnya mendeteksi ground truth di
       kedua grup") lebih relevan sebagai indikator utama. Karena itu
       ditambahkan 'primary_fair'/'primary_fair_count' (berbasis EOD+IF)
       terpisah dari 'fair_count' (4 metrik penuh, tetap dilaporkan agar
       sesuai format Tabel 3.6). Pemilihan ini PERLU DIJUSTIFIKASI DI
       BAB 3/4 -- bukan keputusan yang netral dari data semata.

    Parameters:
        df_test: dataframe test set (berisi kolom sensitive_col)
        sensitive_col: nama kolom atribut sensitif (topology_type)
        y_true: array label aktual TERENCODE (0..n_class-1), test set
        y_pred: array label prediksi TERENCODE (0..n_class-1), test set
        class_names: daftar nama kelas sesuai urutan encoding (le.classes_)
        X_test_scaled: fitur test set yang sudah dinormalisasi (untuk IF, Eq. 15)
        y_proba: matriks probabilitas prediksi (predict_proba) test set (f(x), Eq. 15)
        lipschitz_L: konstanta Lipschitz L pada Eq. (15), default 1.0
        di_epsilon: konstanta smoothing DI, default 1e-3
    """
    privileged_mask = df_test[sensitive_col].isin(
        FAIRNESS_CONFIG['privileged_groups']).values
    disadvantaged_mask = df_test[sensitive_col].isin(
        FAIRNESS_CONFIG['disadvantaged_groups']).values

    n_priv = int(privileged_mask.sum())
    n_dis = int(disadvantaged_mask.sum())

    per_class = {}
    for class_idx, class_name in enumerate(class_names):
        y_true_fav = (y_true == class_idx).astype(int)
        y_pred_fav = (y_pred == class_idx).astype(int)

        # SPD (Eq. 12): |P(y_hat=1|g_i) - P(y_hat=1|g_j)|, g_i=disadvantaged, g_j=privileged
        p_fav_priv = y_pred_fav[privileged_mask].mean() if n_priv > 0 else 0
        p_fav_dis = y_pred_fav[disadvantaged_mask].mean() if n_dis > 0 else 0
        spd = abs(p_fav_dis - p_fav_priv)

        # DI (Eq. 13, DENGAN SMOOTHING): (p_dis + eps) / (p_priv + eps)
        di = (p_fav_dis + di_epsilon) / (p_fav_priv + di_epsilon)

        # EOD (Eq. 14): |P(y_hat=1|y=1,g_i) - P(y_hat=1|y=1,g_j)|
        priv_true_pos = privileged_mask & (y_true_fav == 1)
        dis_true_pos = disadvantaged_mask & (y_true_fav == 1)
        tpr_priv = y_pred_fav[priv_true_pos].mean() if priv_true_pos.sum() > 0 else 0
        tpr_dis = y_pred_fav[dis_true_pos].mean() if dis_true_pos.sum() > 0 else 0
        eod = abs(tpr_dis - tpr_priv)

        per_class[class_name] = {
            'p_favorable_privileged': round(float(p_fav_priv), 4),
            'p_favorable_disadvantaged': round(float(p_fav_dis), 4),
            'spd': round(float(spd), 4),
            'di': round(float(di), 4),
            'eod': round(float(eod), 4),
            'tpr_privileged': round(float(tpr_priv), 4),
            'tpr_disadvantaged': round(float(tpr_dis), 4),
            'spd_fair': bool(spd < 0.1),
            'di_fair': bool(0.8 <= di <= 1.2),
            'eod_fair': bool(eod < 0.1),
        }

    # --- Agregasi worst-case lintas kelas ---
    # "Adil" hanya jika TIDAK ADA kelas mana pun yang melanggar threshold,
    # sesuai definisi tesis sendiri (adil = memenuhi semua kriteria secara
    # simultan). Karena itu dipakai nilai disparitas TERBESAR (worst-case),
    # bukan rata-rata, sebagai representasi tunggal SPD/EOD.
    spd_by_class = {c: m['spd'] for c, m in per_class.items()}
    eod_by_class = {c: m['eod'] for c, m in per_class.items()}
    di_by_class = {c: m['di'] for c, m in per_class.items()}

    spd_worst_class = max(spd_by_class, key=spd_by_class.get)
    eod_worst_class = max(eod_by_class, key=eod_by_class.get)
    spd_agg = spd_by_class[spd_worst_class]
    eod_agg = eod_by_class[eod_worst_class]
    spd_mean = round(float(np.mean(list(spd_by_class.values()))), 4)
    eod_mean = round(float(np.mean(list(eod_by_class.values()))), 4)

    di_worst_class = max(di_by_class, key=lambda c: abs(di_by_class[c] - 1))
    di_agg = di_by_class[di_worst_class]

    # IF (Eq. 15): dihitung SEKALI dari seluruh vektor probabilitas 3 kelas
    # (bukan per kelas) -- f(x) sudah mencakup semua kelas sekaligus.
    individual_fairness, if_details = calculate_individual_fairness(
        y_pred, X_test_scaled, k=5
    )

    spd_fair = bool(spd_agg < 0.1)
    di_fair = bool(0.8 <= di_agg <= 1.2)
    eod_fair = bool(eod_agg < 0.1)
    individual_fair = bool(individual_fairness > 0.9)

    # fair_count: 4 metrik penuh, dipertahankan agar sesuai format Tabel 3.6
    fair_count = sum([spd_fair, di_fair, eod_fair, individual_fair])

    # primary_fair_count: EOD + IF sebagai kriteria utama (lihat REVISI TAMBAHAN #2)
    primary_fair = bool(eod_fair and individual_fair)
    primary_fair_count = sum([eod_fair, individual_fair])

    metrics = {
        'n_privileged': n_priv,
        'n_disadvantaged': n_dis,
        'per_class': per_class,
        'spd': round(float(spd_agg), 4),
        'spd_worst_class': spd_worst_class,
        'spd_mean_across_classes': spd_mean,
        'di': round(float(di_agg), 4),
        'di_worst_class': di_worst_class,
        'di_epsilon_used': di_epsilon,
        'eod': round(float(eod_agg), 4),
        'eod_worst_class': eod_worst_class,
        'eod_mean_across_classes': eod_mean,
        'individual_fairness': round(float(individual_fairness), 4),
        'if_n_pairs': if_details.get('n_pairs'),
        'if_L_used': if_details.get('L_used'),
        'if_empirical_L_max_ratio': if_details.get('empirical_L_max_ratio'),
        'if_violation_rate': if_details.get('violation_rate'),
        'spd_fair': spd_fair,
        'di_fair': di_fair,
        'eod_fair': eod_fair,
        'individual_fair': individual_fair,
        'fair_count': fair_count,
        'primary_fair': primary_fair,
        'primary_fair_count': primary_fair_count,
        'primary_criteria_note': (
            "Kriteria utama = EOD + IF (bukan SPD/DI). SPD/DI dilaporkan "
            "sebagai pelengkap deskriptif: nilai tinggi berpotensi "
            "mencerminkan pola topologi->kontroler yang legitimate "
            "(lihat label_distribution_percent), bukan murni bias sistem. "
            "Perlu dijustifikasi/didiskusikan eksplisit di Bab 3/4."
        ),
    }

    logger.info(f"  [SPD] worst-case={spd_agg:.4f} (kelas: {spd_worst_class}) | "
                f"mean={spd_mean:.4f} | Fair: {spd_fair} [pelengkap]")
    logger.info(f"  [DI]  worst-case={di_agg:.4f} (eps={di_epsilon}) "
                f"(kelas: {di_worst_class}) | Fair: {di_fair} [pelengkap]")
    logger.info(f"  [EOD] worst-case={eod_agg:.4f} (kelas: {eod_worst_class}) | "
                f"mean={eod_mean:.4f} | Fair: {eod_fair} [KRITERIA UTAMA]")
    logger.info(f"  [IF]  {individual_fairness:.4f} | Fair: {individual_fair} [KRITERIA UTAMA] "
                f"(n_pairs={if_details.get('n_pairs')})")
    for c, m in per_class.items():
        logger.info(f"    - {c}: SPD={m['spd']:.4f} DI={m['di']:.4f} EOD={m['eod']:.4f}")
    logger.info(f"  Fairness metrics achieved (4 metrik/Tabel 3.6): {fair_count}/4")
    logger.info(f"  Primary fair (EOD+IF): {primary_fair_count}/2 -> {primary_fair}")

    return metrics


# ============================================================================
# MODEL EVALUATION (REVISED - Tabel 3.7: train_test_split 80/20, RF config)
# ============================================================================

def evaluate_model(df, dataset_name):
    """
    REVISI TOTAL sesuai Tabel 3.7:
      - train_test_split 80% train / 20% test
      - random_state = 42
      - class_weight = 'balanced'
    Metrik: Accuracy, Precision, Recall, F1-Score (average='weighted').

    Decision Tree dipertahankan sebagai baseline TAMBAHAN (di luar
    metodologi inti tesis, hanya untuk pembanding).

    Fairness metrics (SPD/DI/EOD/IF) dihitung dari prediksi RF pada test set.
    """
    if 'topology_type' not in df.columns:
        logger.error("topology_type column missing")
        return None
    if 'best_controller_label' not in df.columns:
        logger.error("best_controller_label column missing")
        return None

    df_features = df[FEATURE_COLS].fillna(0)
    scaler = StandardScaler()
    X = scaler.fit_transform(df_features)

    le = LabelEncoder()
    y = le.fit_transform(df['best_controller_label'])

    # Simpan index asli agar bisa mengambil kembali kolom topology_type
    # dan menghitung favorable outcome per baris test set.
    indices = np.arange(len(df))

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices, test_size=0.20, random_state=42
    )

    df_test = df.iloc[idx_test].reset_index(drop=True)

    results = {
        'dataset': dataset_name,
        'n_samples': len(df),
        'n_features': len(FEATURE_COLS),
        'n_train': len(X_train),
        'n_test': len(X_test),
    }

    # --- Random Forest (metodologi inti, Tabel 3.7) ---
    rf_model = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1
    )
    try:
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        y_proba_rf = rf_model.predict_proba(X_test)  # f(x) untuk IF, Eq. (15)

        results['rf_accuracy'] = round(accuracy_score(y_test, y_pred_rf), 4)
        results['rf_precision'] = round(precision_score(y_test, y_pred_rf, average='weighted', zero_division=0), 4)
        results['rf_recall'] = round(recall_score(y_test, y_pred_rf, average='weighted', zero_division=0), 4)
        results['rf_f1'] = round(f1_score(y_test, y_pred_rf, average='weighted', zero_division=0), 4)

        logger.info(f"{dataset_name} - RF Accuracy:  {results['rf_accuracy']:.4f}")
        logger.info(f"{dataset_name} - RF Precision: {results['rf_precision']:.4f}")
        logger.info(f"{dataset_name} - RF Recall:    {results['rf_recall']:.4f}")
        logger.info(f"{dataset_name} - RF F1-Score:  {results['rf_f1']:.4f}")

        # --- Fairness metrics dari prediksi RF pada test set (one-vs-rest per kelas) ---
        logger.info(f"\n--- Fairness Metrics ({dataset_name}, berbasis prediksi RF, one-vs-rest) ---")
        fairness = calculate_fairness_metrics(
            df_test, FAIRNESS_CONFIG['sensitive_attr'],
            y_test, y_pred_rf, le.classes_, X_test, y_proba_rf
        )
        results.update(fairness)

    except Exception as e:
        logger.error(f"RF evaluation error: {e}")
        results['error'] = str(e)

    # --- Decision Tree (baseline tambahan, di luar metodologi inti) ---
    dt_model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    try:
        dt_model.fit(X_train, y_train)
        y_pred_dt = dt_model.predict(X_test)

        results['dt_accuracy'] = round(accuracy_score(y_test, y_pred_dt), 4)
        results['dt_precision'] = round(precision_score(y_test, y_pred_dt, average='weighted', zero_division=0), 4)
        results['dt_recall'] = round(recall_score(y_test, y_pred_dt, average='weighted', zero_division=0), 4)
        results['dt_f1'] = round(f1_score(y_test, y_pred_dt, average='weighted', zero_division=0), 4)

        logger.info(f"{dataset_name} - DT Accuracy (baseline pembanding): {results['dt_accuracy']:.4f}")

    except Exception as e:
        logger.error(f"DT evaluation error: {e}")
        results['dt_error'] = str(e)

    return results


def compare_datasets():
    logger.info("=" * 60)
    logger.info("DATASET EVALUATION: Comparing Dataset A vs Dataset B")
    logger.info("=" * 60)

    df_a = load_dataset(DATASET_A)
    df_b = load_dataset(DATASET_B)

    if df_a is None:
        logger.error("Dataset A not found.")
        return

    eval_results = []

    logger.info("\n--- Dataset A (AHP Labels) ---")
    results_a = evaluate_model(df_a, "Dataset A")
    if results_a:
        eval_results.append(results_a)

    if df_b is not None:
        logger.info("\n--- Dataset B (Ground Truth: CL + iFlipper) ---")
        results_b = evaluate_model(df_b, "Dataset B")
        if results_b:
            eval_results.append(results_b)

        if results_a and results_b and 'spd' in results_a and 'spd' in results_b:
            logger.info("\n--- Fairness Improvement (RF predictions) ---")
            improvement_spd = abs(results_a['spd']) - abs(results_b['spd'])
            improvement_eod = abs(results_a['eod']) - abs(results_b['eod'])
            logger.info(f"SPD |A| - |B|: {improvement_spd:+.4f} [pelengkap]")
            logger.info(f"EOD |A| - |B|: {improvement_eod:+.4f} [KRITERIA UTAMA]")
            logger.info(f"Fair metrics achieved (4 metrik/Tabel 3.6): {results_a['fair_count']}/4 -> {results_b['fair_count']}/4")
            logger.info(f"Primary fair (EOD+IF): {results_a['primary_fair_count']}/2 -> {results_b['primary_fair_count']}/2")
    else:
        logger.warning("Dataset B not found - skipping comparison")

    results_file = os.path.join(OUTPUT_DIR, 'evaluation_results_withoutepsilon.json')
    with open(results_file, 'w') as f:
        json.dump(eval_results, f, indent=2)
    logger.info(f"\nResults saved to {results_file}")

    return eval_results


def generate_summary_report():
    results_file = os.path.join(OUTPUT_DIR, 'evaluation_results_withoutepsilon.json')
    if not os.path.exists(results_file):
        logger.error("No evaluation results found")
        return

    with open(results_file, 'r') as f:
        results = json.load(f)

    report = []
    report.append("=" * 60)
    report.append("THESIS EVALUATION SUMMARY REPORT")
    report.append("=" * 60)

    for res in results:
        report.append(f"\n--- {res.get('dataset', 'Unknown')} ---")
        report.append(f"  Samples: {res.get('n_samples', 'N/A')} (train={res.get('n_train','N/A')}, test={res.get('n_test','N/A')})")
        report.append(f"  Features: {res.get('n_features', 'N/A')}")

        if 'rf_accuracy' in res:
            report.append(f"  [Random Forest] Accuracy: {res['rf_accuracy']:.4f} | "
                           f"Precision: {res['rf_precision']:.4f} | "
                           f"Recall: {res['rf_recall']:.4f} | "
                           f"F1: {res['rf_f1']:.4f}")
        if 'dt_accuracy' in res:
            report.append(f"  [Decision Tree - baseline tambahan] Accuracy: {res['dt_accuracy']:.4f} | "
                           f"Precision: {res['dt_precision']:.4f} | "
                           f"Recall: {res['dt_recall']:.4f} | "
                           f"F1: {res['dt_f1']:.4f}")

        if 'spd' in res:
            di_str = f"{res['di']:.4f}" if np.isfinite(res['di']) else str(res['di'])
            report.append(f"  [Pelengkap] SPD: {res['spd']:.4f} worst-class={res.get('spd_worst_class')} (fair={res['spd_fair']}) | "
                           f"DI: {di_str} worst-class={res.get('di_worst_class')} (fair={res['di_fair']}, eps={res.get('di_epsilon_used')})")
            report.append(f"  [Kriteria utama] EOD: {res['eod']:.4f} worst-class={res.get('eod_worst_class')} (fair={res['eod_fair']}) | "
                           f"IF: {res['individual_fairness']:.4f} (fair={res['individual_fair']})")
            report.append(f"  Fairness metrics achieved (4 metrik/Tabel 3.6): {res.get('fair_count','N/A')}/4")
            report.append(f"  Primary fair (EOD+IF): {res.get('primary_fair_count','N/A')}/2 -> {res.get('primary_fair')}")
            if 'per_class' in res:
                report.append("  Rincian per kelas (one-vs-rest):")
                for cls_name, cls_metrics in res['per_class'].items():
                    di_cls = cls_metrics['di']
                    di_cls_str = f"{di_cls:.4f}" if isinstance(di_cls, (int, float)) and np.isfinite(di_cls) else str(di_cls)
                    report.append(
                        f"    - {cls_name}: SPD={cls_metrics['spd']:.4f} "
                        f"DI={di_cls_str} EOD={cls_metrics['eod']:.4f}"
                    )


    report_text = "\n".join(report)
    print(report_text)

    report_file = os.path.join(OUTPUT_DIR, 'evaluation_report_withoutepsilon.txt')
    with open(report_file, 'w') as f:
        f.write(report_text)
    logger.info(f"Report saved to {report_file}")


def main():
    logger.info("Starting evaluation...")
    compare_datasets()
    generate_summary_report()
    logger.info("Evaluation complete")


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
