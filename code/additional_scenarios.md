# Additional Scenarios for codeICCRAIDS

## Complete Pipeline Overview

```
codeICCRAIDS.py ──► Dataset A ──► cleaning.py ──► Dataset A_clean ──► Dataset B
     (Gen)        (AHP)       (CL+iFlipper)    (Filtered)      (Fair)
                           │
                           ▼
                    evaluation.py ──► results/
                    visualization.py
```

## Current Scenarios (35 total)

### Star Topologies (7)
| Scenario | Nodes | Bandwidth | Purpose |
|----------|-------|---------|---------|
| Star_4nodes_10M | 4 | 10M | Minimal |
| Star_8nodes_30M | 8 | 30M | Baseline |
| Star_8nodes_100M | 8 | 100M | Standard |
| Star_8nodes_1G | 8 | 1G | High BW |
| Star_12nodes_100M | 12 | 100M | Medium scale |
| Star_16nodes_100M | 16 | 100M | Large scale |
| Star_16nodes_1G | 16 | 1G | Large + High BW |

### Tree Topologies (6)
| Scenario | Depth | Fanout | Bandwidth | Purpose |
|----------|-------|-------|---------|---------|
| Tree_D2_F2_50M | 2 | 2 | 50M | Shallow balanced |
| Tree_D2_F3_50M | 2 | 3 | 50M | Shallow wide |
| Tree_D3_F2_50M | 3 | 2 | 50M | Medium depth |
| Tree_D3_F2_80M | 3 | 2 | 80M | Standard |
| Tree_D3_F2_1G | 3 | 2 | 1G | High BW |
| Tree_D4_F2_100M | 4 | 2 | 100M | Deep tree |

### Linear Topologies (8)
| Scenario | Nodes | Bandwidth | Purpose |
|----------|-------|---------|---------|
| Linear_4nodes_10M | 4 | 10M | Minimal |
| Linear_6nodes_30M | 6 | 30M | Small |
| Linear_8nodes_50M | 8 | 50M | Baseline |
| Linear_8nodes_100M | 8 | 100M | Standard |
| Linear_8nodes_1G | 8 | 1G | High BW |
| Linear_12nodes_50M | 12 | 50M | Medium |
| Linear_12nodes_1G | 12 | 1G | Long + High BW |
| Linear_16nodes_100M | 16 | 100M | Long chain |

### Mesh Topologies (7)
| Scenario | Nodes | Bandwidth | Purpose |
|----------|-------|---------|---------|
| Mesh_4nodes_10M | 4 | 10M | Minimal |
| Mesh_4nodes_100M | 4 | 100M | Small dense |
| Mesh_6nodes_30M | 6 | 30M | Baseline |
| Mesh_6nodes_100M | 6 | 100M | Standard |
| Mesh_6nodes_1G | 6 | 1G | High BW |
| Mesh_8nodes_50M | 8 | 50M | Medium |
| Mesh_8nodes_5M | 8 | 5M | Stress test |

### Ring Topologies (7)
| Scenario | Nodes | Bandwidth | Purpose |
|----------|-------|---------|---------|
| Ring_4nodes_10M | 4 | 10M | Minimal |
| Ring_6nodes_30M | 6 | 30M | Small |
| Ring_8nodes_50M | 8 | 50M | Baseline |
| Ring_8nodes_1G | 8 | 1G | High BW |
| Ring_10nodes_60M | 10 | 60M | Standard |
| Ring_10nodes_1G | 10 | 1G | Large + High BW |
| Ring_12nodes_100M | 12 | 100M | Large |
| Ring_16nodes_80M | 16 | 80M | XL Ring |

## Dataset Distribution for Fairness

| Group | Topologies | Scenarios | % of Total |
|-------|-----------|----------|-----------|
| Privileged | Star, Tree, Linear | 21 | 60% |
| Disadvantaged | Mesh, Ring | 14 | 40% |

This balance supports your **iFlipper** fairness correction.

## Iteration Support

- **ITERATIONS = 3** (configurable in codeICCRAIDS.py)
- Each scenario runs 3 times per controller
- Total samples: 35 scenarios × 3 iterations × 3 controllers = **315 samples**

## Supplementary Scripts

| Script | Purpose | Output |
|--------|---------|-------|
| `cleaning.py` | Confident Learning + iFlipper | Dataset B |
| `evaluation.py` | Model comparison (RF/DT) | Evaluation metrics |
| `visualization.py` | Charts for thesis | PNG + TXT reports |

## Execution Order

```bash
# 1. Generate Dataset A
python3 codeICCRAIDS.py

# 2. Clean & Fairness (CL + iFlipper)
python3 cleaning.py

# 3. Evaluate
python3 evaluation.py

# 4. Visualize
python3 visualization.py
```

## Results Location

All outputs saved to:
- `data/sdn_dataset_ahp.csv` (Dataset A)
- `data/sdn_dataset_clean.csv` (Dataset A_clean)
- `data/sdn_dataset_fair.csv` (Dataset B)
- `results/` (evaluation & visualization)