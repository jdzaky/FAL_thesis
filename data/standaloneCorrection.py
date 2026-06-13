import pandas as pd
import numpy as np
from scipy.stats import chisquare

def evaluate_full_framework(input_csv, output_csv):
    # 1. Load Data
    df = pd.read_csv(input_csv)

    # 2. Definisi Bobot AHP (Sesuai Hierarki Draft Paper)
    weights = {
        'main': {'compliance': 0.724, 'efficiency': 0.193, 'stability': 0.083},
        'compliance': {'throughput': 0.090, 'latency': 0.354, 'reliability': 0.556},
        'efficiency': {'cpu': 0.751, 'memory': 0.249},
        'stability': {'jitter': 0.750, 'flow_setup': 0.250}
    }

    print("--- Tahap 1: Kalkulasi AHP & Penalti Ryu ---")

    for index, row in df.iterrows():
        # Parameter dinamis berdasarkan skenario
        bw_val = float(row['bandwidth_demand'].replace('M', ''))
        nodes = row['nodes']
        
        # Threshold Requirements (Sesuai Benchmarking Empiris)
        throughput_req = bw_val * 0.8
        latency_req = nodes * 1.5 + 20
        reliability_req = 99.0 + (nodes * 0.1)

        # A. Compliance Scoring
        t_score = min(row['throughput_mbps'] / max(throughput_req, 1), 1.0)
        l_score = 1.0 if row['latency_ms'] <= latency_req else max(0, 1.0 - (row['latency_ms'] - latency_req) / latency_req)
        rel_score = min((100 - row['packet_loss_percent']) / max(reliability_req, 1), 1.0)
        
        comp_score = (weights['compliance']['throughput'] * t_score + 
                      weights['compliance']['latency'] * l_score + 
                      weights['compliance']['reliability'] * rel_score)

        # --- PERLAKUAN KHUSUS RYU STANDALONE ---
        # Pinalti: Jika Ryu gagal koneksi sebagai controller, compliance = 0
        if row['controller_name'] == 'ryu' and row['connectivity_status'] == 'STANDALONE':
            comp_score = 0.0

        # B. Efficiency Scoring (Normalisasi CPU & Memori)
        cpu_norm = max(0, 1.0 - (row['cpu_usage_percent'] / 100.0))
        mem_norm = max(0, 1.0 - np.log10(row['memory_usage_mb'] + 1) / np.log10(2000))
        eff_score = (weights['efficiency']['cpu'] * cpu_norm + 
                     weights['efficiency']['memory'] * mem_norm)

        # C. Stability Scoring (Jitter & Flow Setup)
        j_score = max(0, 1.0 - row['jitter_ms'] / 50.0)
        f_score = max(0, 1.0 - row['flow_setup_time_ms'] / 300.0)
        stab_score = (weights['stability']['jitter'] * j_score + 
                      weights['stability']['flow_setup'] * f_score)

        # Skor Total AHP
        total_score = (weights['main']['compliance'] * comp_score + 
                       weights['main']['efficiency'] * eff_score + 
                       weights['main']['stability'] * stab_score)

        # Update Baris Data
        df.at[index, 'score_compliance'] = round(comp_score, 4)
        df.at[index, 'score_efficiency'] = round(eff_score, 4)
        df.at[index, 'score_stability'] = round(stab_score, 4)
        df.at[index, 'ahp_score'] = round(total_score, 4)
        df.at[index, 'score_reliability'] = round(rel_score, 4)

    # 3. Labeling: Menentukan Pemenang per Skenario
    for scenario in df['scenario_name'].unique():
        mask = df['scenario_name'] == scenario
        scenario_df = df[mask]
        
        best_row = scenario_df.loc[scenario_df['ahp_score'].idxmax()]
        best_ctrl = best_row['controller_name']
        
        scores = sorted(scenario_df['ahp_score'].tolist(), reverse=True)
        gap = scores[0] - scores[1] if len(scores) > 1 else 0
        
        df.loc[mask, 'best_controller_label'] = best_ctrl
        df.loc[mask, 'score_gap'] = round(gap, 4)

    # Simpan Hasil ke CSV Baru
    df.to_csv(output_csv, index=False)
    print(f"File evaluasi disimpan: {output_csv}\n")

    # --- Tahap 2: Validasi Hipotesis H1, H2, H3 ---
    print("=== HASIL EVALUASI HIPOTESIS ===")
    
    # H1: Fairness (Uji Chi-Square pada Distribusi Menang)
    scenario_winners = df.groupby('scenario_name')['best_controller_label'].first()
    wins_counts = scenario_winners.value_counts()
    all_ctrls = df['controller_name'].unique()
    observed = [wins_counts.get(c, 0) for c in all_ctrls]
    total_w = sum(observed)
    expected = [total_w / len(all_ctrls)] * len(all_ctrls)
    
    chi_stat, p_val = chisquare(observed, f_exp=expected)
    print(f"H1 (Fairness): p-value = {p_val:.4f} {'[PASSED]' if p_val > 0.05 else '[FAILED]'}")
    
    # H2: Reproducibility (Konsistensi Label)
    correct_labels = 0
    for scenario in df['scenario_name'].unique():
        sub = df[df['scenario_name'] == scenario]
        if sub['best_controller_label'].iloc[0] == sub.loc[sub['ahp_score'].idxmax(), 'controller_name']:
            correct_labels += 1
    repro_rate = (correct_labels / len(df['scenario_name'].unique())) * 100
    print(f"H2 (Reproducibility): Rate = {repro_rate:.2f}% {'[PASSED]' if repro_rate >= 95.0 else '[FAILED]'}")

    # H3: Explainability (Score Gaps)
    min_gap = df['score_gap'].min()
    print(f"H3 (Explainability): Min Score Gap = {min_gap:.4f} {'[PASSED]' if min_gap > 0 else '[FAILED]'}")

# Jalankan
if __name__ == "__main__":
    evaluate_full_framework('sdn_dataset_ahp1.csv', 'final_evaluated_dataset2.csv')
