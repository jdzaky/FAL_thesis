"""
feature_importance_proxy_bias.py
Bukti Lapis 2: Feature Importance RF Dataset A vs Dataset B
untuk sub-bagian 4.7 tesis (Bukti Bias Proksi Topologi)

Konfigurasi disesuaikan dengan Tabel 3.7 (Bab III):
- random_state = 42
- n_estimators  = 100
- validasi      = 5-fold StratifiedKFold
- n_jobs        = -1
(class_weight='balanced' DIHAPUS agar identik dengan model utama
di 3.2.5 / evaluation_revised.py — Tabel 3.7 tidak mencantumkannya)

Partisi fold memakai StratifiedKFold(random_state=42) yang sama
dengan skema di evaluation_revised.py, sehingga fold pada analisis
feature importance ini identik dengan fold yang menghasilkan
Tabel 4.8 (F1 per fold). Feature importance (MDI) dihitung per
fold lalu diagregasi menjadi mean +/- std, mengikuti format yang
sama dengan Tabel 4.4.1.

CATATAN: StratifiedKFold dipanggil dengan shuffle=False (default),
TANPA random_state -- ini disengaja, mengikuti evaluation_revised.py.
Sklearn menolak/mengabaikan random_state ketika shuffle=False, sehingga
random_state tidak diberikan di sini. Reprodusibilitas partisi fold
tidak bergantung pada seed, melainkan pada urutan baris PATH_A/PATH_B
yang tetap (deterministik) setiap kali script dijalankan -- sehingga
fold pada analisis feature importance ini identik dengan fold yang
menghasilkan Tabel 4.8 (F1 per fold), selama urutan baris CSV sumber
tidak diubah.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

FEATURE_COLS = ['throughput_mbps', 'latency_ms', 'packet_loss_percent',
                'jitter_ms', 'cpu_usage_percent', 'memory_usage_mb',
                'flow_setup_time_ms', 'nodes']

PATH_A = 'data/final_sdn_dataset_ahp.csv'
PATH_B = 'data/GROUND_TRUTH.csv'

RANDOM_STATE = 42
N_SPLITS = 5


def get_feature_importance_cv(df, label='Dataset'):
    """
    Menghitung feature importance (Mean Decrease in Impurity) RF
    per fold menggunakan 5-fold StratifiedKFold (random_state=42),
    lalu mengagregasi hasil menjadi mean +/- std per fitur.

    RF di-fit HANYA pada training fold di tiap iterasi (bukan pada
    100% data), agar konsisten dengan skema validasi Tabel 3.7 dan
    partisi fold yang dipakai untuk paired t-test F1-weighted.
    """
    le = LabelEncoder()
    X = df[FEATURE_COLS].values
    y = le.fit_transform(df['best_controller_label'].values)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=False)

    fold_importances = []   # satu pd.Series per fold
    nodes_ranks = []        # rank 'nodes' di tiap fold (1 = paling penting)

    for train_idx, _ in skf.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]

        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)

        imp = pd.Series(rf.feature_importances_, index=FEATURE_COLS)
        fold_importances.append(imp)

        rank = imp.sort_values(ascending=False).index.get_loc('nodes') + 1
        nodes_ranks.append(rank)

    imp_matrix = pd.DataFrame(fold_importances)  # baris = fold, kolom = fitur
    imp_mean = imp_matrix.mean().sort_values(ascending=False)
    imp_std = imp_matrix.std().reindex(imp_mean.index)

    print(f"\n=== Feature Importance (5-fold CV, mean +/- std) — {label} ===")
    print(f"{'Fitur':<25} {'Mean':>8} {'Std':>8}  {'Rank':>5}")
    print("-" * 52)
    for rank, feat in enumerate(imp_mean.index, 1):
        marker = " <-- PROKSI TOPOLOGI" if feat == 'nodes' else ""
        print(f"{feat:<25} {imp_mean[feat]:>8.4f} {imp_std[feat]:>8.4f}  {rank:>5}{marker}")

    print(f"\nRank 'nodes' per fold: {nodes_ranks}")
    print(f"Rank 'nodes' rata-rata: {np.mean(nodes_ranks):.1f} "
          f"(std={np.std(nodes_ranks):.2f})")

    return imp_mean, imp_std, nodes_ranks


def compare_importance(imp_a, imp_b, ranks_a, ranks_b):
    print("\n=== Perbandingan Feature Importance: Dataset A vs Dataset B ===")
    print(f"{'Fitur':<25} {'A (mean)':>10} {'B (mean)':>10} {'Selisih':>10}")
    print("-" * 60)
    for feat in FEATURE_COLS:
        val_a = imp_a[feat]
        val_b = imp_b[feat]
        delta = val_b - val_a
        marker = " <--" if feat == 'nodes' else ""
        print(f"{feat:<25} {val_a:>10.4f} {val_b:>10.4f} {delta:>+10.4f}{marker}")

    rank_a = list(imp_a.index).index('nodes') + 1
    rank_b = list(imp_b.index).index('nodes') + 1
    pct_change = ((imp_b['nodes'] - imp_a['nodes']) / imp_a['nodes']) * 100

    print(f"\nRank 'nodes' (berdasarkan mean): Dataset A = #{rank_a}, Dataset B = #{rank_b}")
    print(f"Perubahan importance 'nodes' (mean): {pct_change:+.1f}%")
    print(f"Stabilitas rank 'nodes' per fold — Dataset A: {ranks_a}")
    print(f"Stabilitas rank 'nodes' per fold — Dataset B: {ranks_b}")


def main():
    df_a = pd.read_csv(PATH_A)
    df_b = pd.read_csv(PATH_B)

    imp_a, std_a, ranks_a = get_feature_importance_cv(
        df_a, "Dataset A (AHP — sebelum pipeline)")
    imp_b, std_b, ranks_b = get_feature_importance_cv(
        df_b, "Dataset B (Ground Truth — setelah pipeline)")

    compare_importance(imp_a, imp_b, ranks_a, ranks_b)


if __name__ == '__main__':
    main()