# Thesis Pipeline Summary

## Overview

This directory contains the complete pipeline for your thesis: **Fair Automatic Labeling (FAL) + Confident Learning + iFlipper for SDN Controller Selection**.

---

## Pipeline Flow

```
codeICCRAIDS.py ──► Dataset A ──► cleaning.py ──► Dataset B ──► evaluation.py + visualization.py
   (Generate)     (AHP)       (CL+iFlipper)     (Fair)           (Results)
```

---

## Files Created

### 1. codeICCRAIDS.py
**Purpose:** Generate Dataset A (AHP-based labeled SDN controller performance data)

| Config | Value |
|--------|-------|
| Scenarios | 35 |
| Iterations | 3 |
| Controllers | Ryu, ONOS, Floodlight |
| Topologies | Star, Tree, Linear, Mesh, Ring |
| Total Samples | 315 |

**Scenarios included:**
- Star: 4, 8, 12, 16 nodes (10M to 1G bandwidth)
- Tree: depth 2-4, fanout 2-3 (50M to 1G)
- Linear: 4-16 nodes (10M to 1G)
- Mesh: 4-8 nodes (5M to 1G)
- Ring: 4-16 nodes (10M to 1G)

### 2. cleaning.py
**Purpose:** Post-processing with Confident Learning + iFlipper

**Stages:**
1. **Confident Learning** - Detect noisy labels based on confidence scores
2. **iFlipper** - Apply individual fairness corrections
3. **Output** - Dataset A_clean and Dataset B

### 3. evaluation.py
**Purpose:** Compare Dataset A vs Dataset B

**Metrics:**
- Random Forest accuracy (with cross-validation)
- Decision Tree accuracy
- Statistical Parity Difference
- Equal Opportunity Difference
- Accuracy Gap (privileged vs disadvantaged)

### 4. visualization.py
**Purpose:** Generate thesis charts

**Charts generated:**
- controller_performance.png (bar charts of all metrics)
- ahp_scores_topology.png (AHP by topology)
- fairness_gap.png (privileged vs disadvantaged gap)
- iteration_stability.png (iteration variance)
- scenario_heatmap.png (scenario vs controller)
- dataset_distribution.png (pie charts)
- dataset_summary.txt (text-based report)

---

## Fairness Configuration

From your thesis plan in ringkasanChat.md:

| Attribute | Value |
|----------|-------|
| Sensitive Attribute | topology_type |
| Privileged Group | Linear, Star, Tree |
| Disadvantaged Group | Mesh, Ring |
| AHP Compliance Weight | 0.724 |
| AHP Efficiency Weight | 0.193 |
| AHP Stability Weight | 0.083 |

---

## Dataset Distribution

| Category | Topologies | Count | Percentage |
|----------|------------|-------|------------|
| Privileged | Star, Tree, Linear | 21 | 60% |
| Disadvantaged | Mesh, Ring | 14 | 40% |

This balanced distribution supports **iFlipper** fairness correction.

---

## Execution Commands

```bash
# Step 1: Generate Dataset A (requires Docker + Mininet)
python3 codeICCRAIDS.py

# Step 2: Apply cleaning and fairness
python3 cleaning.py

# Step 3: Evaluate models
python3 evaluation.py

# Step 4: Generate visualizations
python3 visualization.py
```

---

## Output Files

| File | Location | Description |
|------|----------|-------------|
| sdn_dataset_ahp.csv | data/ | Dataset A (original) |
| sdn_dataset_clean.csv | data/ | Dataset A_clean (noise filtered) |
| sdn_dataset_fair.csv | data/ | Dataset B (fair labels) |
| controller_performance.png | results/ | Performance comparison |
| ahp_scores_topology.png | results/ | AHP by topology |
| fairness_gap.png | results/ | Fairness gap chart |
| scenario_heatmap.png | results/ | Scenario heatmap |
| dataset_summary.txt | results/ | Text-based summary |
| evaluation_results.json | results/ | Evaluation metrics |
| evaluation_report.txt | results/ | Final report |

---

## Research Questions Addressed

1. **Does controller performance vary by topology?**
   - Tested across Star, Tree, Linear, Mesh, Ring

2. **Is there fairness bias between privileged vs disadvantaged groups?**
   - Mesh/Ring (disadvantaged) vs Linear/Star/Tree (privileged)

3. **Does high bandwidth (1G) affect fairness?**
   - Included 1G scenarios for all topologies

4. **How stable are the results across iterations?**
   - 3 iterations per scenario for statistical significance

---

## Expected Results

| Stage | Samples | Description |
|-------|---------|-------------|
| Original (Dataset A) | 315 | AHP-labeled data |
| After CL (A_clean) | <315 | Noise filtered |
| After iFlipper (B) | 315 | Fair corrected |

**Fairness improvement target:**
- Statistical Parity Difference: closer to 0
- Accuracy Gap: reduced between groups

---

## How This Supports Your Thesis

From ringkasanChat.md pipeline:

| Tahap | Input | Output | Script |
|-------|-------|--------|--------|
| 1. FAL Generation | Mininet simulation | Dataset A | codeICCRAIDS.py |
| 2. Pre-processing | Dataset A | Ready for CL | cleaning.py |
| 3. Confident Learning | Dataset A | A_clean | cleaning.py |
| 4. iFlipper | A_clean | Dataset B | cleaning.py |
| 5. Model Evaluation | Dataset A & B | Metrics | evaluation.py |
| Visualization | Dataset A | Charts | visualization.py |

---

## Requirements

- Python 3.7+
- Docker (for SDN controllers)
- Mininet (for network simulation)
- Libraries: numpy, pandas, scikit-learn, matplotlib, seaborn

---

## Notes

- Run `codeICCRAIDS.py` first to generate data (requires Docker + Mininet)
- `cleaning.py`, `evaluation.py`, `visualization.py` can be run after data exists
- All scripts output to `data/` and `results/` directories
- Check `logs/generation_ahp.log` for execution details