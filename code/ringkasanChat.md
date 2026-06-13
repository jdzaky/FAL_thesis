# Ringkasan: Pipeline FAL + Confident Learning + iFlipper untuk Thesis

## Tujuan Utama
Mengembangkan pipeline Fair Automatic Labeling (FAL) yang menggabungkan:
- **AHP-based labeling** (dari PreliminaryFALPaper)
- **Confident Learning** untuk deteksi noise/error pada label
- **iFlipper** untuk koreksi individual fairness violations

Tujuan: Membuat data training yang adil (fair) untuk model ML pemilihan SDN controller.

---

## Tahap Pipeline

| Tahap | Nama | Input | Output |
|-------|------|-------|--------|
| 1 | **FAL Data Generation** | Data simulasi Mininet | Dataset A (label AHP) |
| 2 | **Pre-processing** | Dataset A | Siap untuk CL/iFlipper |
| 3 | **Confident Learning** | Dataset A | Dataset A_clean (filter noise) |
| 4 | **iFlipper** | Dataset A_clean | Dataset B (label adil) |
| 5 | **Model Evaluation** | Dataset A & B | Metrics + Kesimpulan |

---

## Detail Setiap Tahap

### Tahap 1: FAL Data Generation (Selesai - dari PreliminaryFALPaper)
- **Simulasi:** 5 topologi (Linear, Mesh, Ring, Star, Tree) × 3 controller (Ryu, ONOS, Floodlight)
- **Metrik:** 7 performa (throughput, latency, packet loss, jitter, CPU, memory, flow setup time)
- **AHP Weights:** Compliance=0.724, Efficiency=0.193, Stability=0.083
- **Fairness Penalty:** Jika 12 ping gagal → Compliance=0
- **Output:** `final_evaluated_dataset2.csv` (Dataset A)

### Tahap 2: Pre-processing
- Konversi label ke format biner (One-vs-All): `is_ryu`, `is_onos`, `is_floodlight`
- Definisi atribut sensitif: **topology_type**
  - Privileged: Linear, Star, Tree
  - Disadvantaged: Mesh, Ring
- Normalisasi fitur numerik

### Tahap 3: Confident Learning
- Bagi dataset berdasarkan grup topologi (Privileged vs Disadvantaged)
- Latih classifier pada masing-masing subset
- Hitung confidence score dan threshold
- Filter noisy labels (off-diagonal elements)
- **Library:** `cleanlab`

### Tahap 4: iFlipper (Individual Fairness)
1. **Similarity Matrix:** Hitung jarak Euclidean antar sampel
2. **Linear Programming:** Minimalkan jumlah label flip dengan constraint
3. **Adaptive Rounding:** Konversi solusi fraksional ke biner
4. **Reverse Greedy:** Opsional -embalikan label jika memungkinkan
- **Library:** `scipy.optimize.linprog` atau `mosek`

### Tahap 5: Model Evaluation
- Latih Random Forest/Decision Tree dengan Dataset A dan B
- Evaluasi:
  - **Accuracy** (akurasi prediksi)
  - **Fairness Metrics:** Statistical Parity Difference, Equal Opportunity Difference
  - **XAI:** SHAP/LIME untuk explainability
- **Statistical Test:** Paired t-test, Wilcoxon signed-rank

---

## Komponen Kunci

| Komponen | Nilai/Detail |
|---------|-------------|
| Atribut Sensitif | topology_type |
| Privileged Group | Linear, Star, Tree |
| Disadvantaged Group | Mesh, Ring |
| AHP Compliance | 0.724 |
| AHP Efficiency | 0.193 |
| AHP Stability | 0.083 |
| Consistency Ratio | 0.047 |
| Ping Failure Threshold | 12 consecutive |

---

## Library yang Direkomendasikan

| Tahap | Library |
|-------|---------|
| FAL Generation | Custom Python (dari PreliminaryFALPaper) |
| Confident Learning | `cleanlab` |
| LP Solver | `scipy.optimize.linprog` atau `mosek` |
| Fairness Metrics | `AIF360` atau `Fairlearn` |
| XAI/Explainability | `SHAP`, `LIME` |
| ML Models | `scikit-learn` (Random Forest, Decision Tree) |

---

## Visual Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PIPELINE FAL + CL + iFlipper                         │
├──────────────────────────────────────────────────────────────────────────────┤
│  Dataset A (AHP)  ──►  Confident Learning  ──►  iFlipper  ──►  Dataset B │
│  (Ground Truth)       (Filter Noise)           (Fair Correction)        │
├─────────────────────────────────────────────────────────────────────────┤
│  Evaluation: Dataset A vs Dataset B                                     │
│  - Accuracy comparison (t-test)                                     │
│  - Fairness metrics (Statistical Parity, Equal Opportunity)            │
│  - Explainability (SHAP/LIME)                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Kesimpulan
Pipeline ini mengabungkan kekuatan dari tiga pendekatan:
1. **FAL** - menghasilkan ground truth yang objektif berbasis AHP
2. **Confident Learning** - mendeteksi dan menyaring label yang noisy
3. **iFlipper** - mengoreksi individual fairness violations

Hasil yang diharapkan: Dataset B yang lebih adil (fair) untuk training ML dengan akurasi yang tetap kompetitif dibandingkan Dataset A.