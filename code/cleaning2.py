#!/usr/bin/env python3
"""
cleaning.py - Fair Confident Learning + iFlipper Implementation

Based on:
- CL.txt: "Mitigating Label Bias in Machine Learning: Fairness through Confident Learning"
- iFlipper/experiments_demo/iFlipper_Demo.ipynb: Individual Fairness through label flipping
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

DATASET_A = os.path.join(DATA_DIR, 'sdn_dataset_ahp.csv')
DATASET_A_CLEAN = os.path.join(DATA_DIR, 'sdn_dataset_clean.csv')
DATASET_B = os.path.join(DATA_DIR, 'sdn_dataset_fair.csv')

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
# CONFIDENT LEARNING IMPLEMENTATION (Based on CL.txt)
# ============================================================================

def psi(x):
    """Eq. (4): ψ(x) = log(1 + x + x²/2)"""
    return np.log(1 + x + (x ** 2) / 2)


def compute_threshold(probs, labels, n_classes, Ns, nu):
    """
    Eq. (5): Compute threshold µ_j using lower bound confidence interval
    - probs: (N, k) predicted probabilities
    - labels: (N,) observed labels
    - Ns: estimated number of clean instances
    - nu: variance parameter
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

        thresholds[j] = mu_j
    return thresholds


def assign_true_labels(probs_A, probs_B, labels, thresh_A, thresh_B, n_classes):
    """
    Algorithm 1 Line 7-11: Assign true labels with conditional check
    - Check θA first, fallback to θB
    """
    N = len(labels)
    z_hat = np.full(N, -1, dtype=int)
    source = np.full(N, '', dtype=object)

    for n in range(N):
        y_n = labels[n]
        p_A = probs_A[n, y_n]
        p_B = probs_B[n, y_n]

        if p_A >= thresh_A.get(y_n, 0):
            z_hat[n] = int(np.argmax(probs_A[n]))
            source[n] = 'A'
        elif p_B >= thresh_B.get(y_n, 0):
            z_hat[n] = int(np.argmax(probs_B[n]))
            source[n] = 'B'
        else:
            source[n] = 'none'

    return z_hat, source


def compute_confident_joint(labels, z_hat, n_classes):
    """
    Eq. (1): Compute confident joint C_bar
    Normalize count matrix per row
    """
    C = np.zeros((n_classes, n_classes), dtype=float)
    for n in range(len(labels)):
        if z_hat[n] == -1:
            continue
        y = labels[n]
        z = z_hat[n]
        C[y, z] += 1

    C_bar = np.zeros_like(C)
    for j in range(n_classes):
        row_sum = C[j].sum()
        n_j = np.sum(labels == j)
        if row_sum > 0:
            C_bar[j] = (C[j] / row_sum) * n_j

    return C_bar


def get_off_diagonal_indices(labels, z_hat):
    """
    Algorithm 1 Line 13: Get instances where y ≠ z (noisy labels)
    """
    biased_idx = []
    for n in range(len(labels)):
        if z_hat[n] != -1 and labels[n] != z_hat[n]:
            biased_idx.append(n)
    return set(biased_idx)


def fair_confident_learning(df, epochs=50, Ns_fraction=0.6, nu=1e-2):
    """
    Algorithm 1 from CL.txt: Fairness through Confident Learning
    
    Parameters:
        df: input dataframe
        epochs: number of training epochs
        Ns_fraction: fraction of estimated clean instances
        nu: variance parameter
    """
    logger.info("=" * 60)
    logger.info("FAIR CONFIDENT LEARNING (CL.txt Algorithm 1)")
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
    logger.info(f"Privileged (A): {len(idx_B)} | Disadvantaged (B): {len(idx_A)}")
    logger.info(f"Estimated clean: {Ns}")
    
    clf_A = LogisticRegression(max_iter=1000, random_state=42)
    clf_B = LogisticRegression(max_iter=1000, random_state=42)
    
    all_biased = set()
    
    for epoch in range(epochs):
        X_A, y_A = X[idx_A], y[idx_A]
        X_B, y_B = X[idx_B], y[idx_B]
        
        clf_A.fit(X_A, y_A)
        clf_B.fit(X_B, y_B)

        all_classes = np.arange(n_classes)

        def predict_proba_full(clf, X_input, all_classes):
            raw = clf.predict_proba(X_input)
            full = np.zeros((len(X_input), len(all_classes)))
            for i, c in enumerate(clf.classes_):
                full[:, c] = raw[:, i]
            return full

        probs_A = predict_proba_full(clf_A, X, all_classes)
        probs_B = predict_proba_full(clf_B, X, all_classes)
        
        thresh_A = compute_threshold(probs_A, y, n_classes, Ns, nu)
        thresh_B = compute_threshold(probs_B, y, n_classes, Ns, nu)
        
        z_hat, source = assign_true_labels(probs_A, probs_B, y, thresh_A, thresh_B, n_classes)
        
        C_bar = compute_confident_joint(y, z_hat, n_classes)
        
        biased_local = get_off_diagonal_indices(y, z_hat)
        all_biased.update(biased_local)
        
        clean_local = [i for i in range(N) if i not in biased_local]
        
        clf_f = LogisticRegression(max_iter=1000, random_state=42)
        if len(clean_local) > 0:
            clf_f.fit(X[clean_local], y[clean_local])
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Noisy detected: {len(all_biased):4d} | "
                  f"Clean: {len(clean_local):4d}")
    
    df_copy['is_noisy_cl'] = False
    df_copy.loc[list(all_biased), 'is_noisy_cl'] = True
    
    noisy_count = len(all_biased)
    logger.info(f"\nConfident Learning detected {noisy_count} noisy samples ({noisy_count/N*100:.1f}%)")
    
    return df_copy


# ============================================================================
# IFLIPPER IMPLEMENTATION (Based on iFlipper_Demo.ipynb)
# ============================================================================

def generate_similarity_matrix(X, method='knn', k=5, threshold=3.0):
    """
    Generate similarity matrix using KNN or threshold method
    Based on iFlipper_Demo.ipynb: generate_sim_matrix
    """
    n = len(X)
    
    if method == 'knn':
        nn = NearestNeighbors(n_neighbors=min(k+1, n), algorithm='ball_tree')
        nn.fit(X)
        _, distances = nn.kneighbors(X)
        
        W = np.zeros((n, n))
        for i in range(n):
            for j in range(1, min(k+1, n)):
                W[i, distances[i, j]] = 1.0
        
        W = (W + W.T) / 2
        np.fill_diagonal(W, 0)
        
    else:
        from sklearn.metrics import pairwise_distances
        dist_matrix = pairwise_distances(X)
        
        W = np.zeros((n, n))
        W[dist_matrix <= threshold] = 1.0
        W = W - np.eye(n)
    
    return W


def measure_error(y, edge, w_edge):
    """
    Measure total error: sum of weighted mismatches
    """
    return np.sum(w_edge * (y[edge[:, 0]] != y[edge[:, 1]]))


def iflipper_transform(y, W, error_budget):
    """
    iFlipper: Individual Fairness through Label Flipping
    
    Based on iFlipper_Demo.ipynb:
    - Minimize label flips with similarity constraint
    - Linear programming approach
    """
    n = len(y)
    
    edge = np.array(np.triu(np.ones((n, n)) == 1).nonzero()).T
    edge = edge[edge[:, 0] < edge[:, 1]]
    
    w_edge = W[edge[:, 0], edge[:, 1]]
    
    valid_edges = w_edge > 0
    edge = edge[valid_edges]
    w_edge = w_edge[valid_edges]
    
    m = len(edge)
    
    y_flipped = y.copy()
    
    edge_errors = np.array([y[edge[i, 0]] != y[edge[i, 1]] for i in range(m)])
    current_error = np.sum(w_edge * edge_errors)
    
    if current_error <= error_budget:
        logger.info(f"iFlipper: Error {current_error:.1f} within budget {error_budget:.1f}")
        return y_flipped
    
    flip_candidates = []
    flip_impacts = []
    
    for idx in range(n):
        potential_flipped = y.copy()
        potential_flipped[idx] = 1 - potential_flipped[idx]
        
        new_errors = np.array([potential_flipped[edge[i, 0]] != potential_flipped[edge[i, 1]] for i in range(m)])
        new_error = np.sum(w_edge * new_errors)
        
        reduction = current_error - new_error
        flip_candidates.append(idx)
        flip_impacts.append(reduction)
    
    flip_impacts = np.array(flip_impacts)
    sorted_indices = np.argsort(-flip_impacts)
    
    for idx in sorted_indices:
        if flip_impacts[idx] > 0:
            y_flipped[flip_candidates[idx]] = 1 - y_flipped[flip_candidates[idx]]
            
            edge_errors = np.array([y_flipped[edge[i, 0]] != y_flipped[edge[i, 1]] for i in range(m)])
            current_error = np.sum(w_edge * edge_errors)
            
            if current_error <= error_budget:
                break
    
    flips = np.sum(y != y_flipped)
    logger.info(f"iFlipper: {flips} labels flipped, error now {current_error:.1f}")
    
    return y_flipped


def apply_iflipper_fairness(df, error_fraction=0.1):
    """
    Apply iFlipper algorithm for individual fairness
    
    Based on iFlipper_Demo.ipynb:
    - Generate similarity matrix from features
    - Optimize label flips within error budget
    """
    logger.info("=" * 60)
    logger.info("IFLIPPER: Individual Fairness (iFlipper_Demo.ipynb)")
    logger.info("=" * 60)
    
    feature_cols = ['throughput_mbps', 'latency_ms', 'packet_loss_percent', 
                   'jitter_ms', 'cpu_usage_percent', 'memory_usage_mb',
                   'flow_setup_time_ms', 'nodes']
    
    df_fair = df.copy()
    df_features = df_fair[feature_cols].fillna(0)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(df_features)
    
    le = LabelEncoder()
    y_original = le.fit_transform(df_fair['best_controller_label'])
    
    logger.info(f"Generating similarity matrix for {len(X)} samples...")
    W = generate_similarity_matrix(X, method='knn', k=5)
    
    edge = np.array(np.triu(np.ones((len(X), len(X))) == 1).nonzero()).T
    edge = edge[edge[:, 0] < edge[:, 1]]
    w_edge = W[edge[:, 0], edge[:, 1]]
    
    valid_edges = w_edge > 0
    edge = edge[valid_edges]
    w_edge = w_edge[valid_edges]
    
    initial_error = measure_error(y_original, edge, w_edge)
    error_budget = initial_error * error_fraction
    
    logger.info(f"Initial error: {initial_error:.1f}")
    logger.info(f"Error budget: {error_budget:.1f} ({error_fraction*100}%)")
    
    df_fair['iflipper_applied'] = False
    
    for target_class in range(len(le.classes_)):
        y_binary = (y_original == target_class).astype(int)
        
        y_flipped = iflipper_transform(y_binary, W, error_budget)
        
        flipped_indices = np.where(y_binary != y_flipped)[0]
        
        if len(flipped_indices) > 0:
            df_fair.loc[flipped_indices, 'best_controller_label'] = le.classes_[target_class]
            df_fair.loc[flipped_indices, 'iflipper_applied'] = True
    
    total_flips = df_fair['iflipper_applied'].sum()
    logger.info(f"iFlipper: {total_flips} labels modified for fairness")
    
    df_fair = df_fair.drop(columns=['iflipper_applied'], errors='ignore')
    
    return df_fair


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def apply_confident_learning_filter(df):
    """Apply Confident Learning noise detection"""
    df_cl = fair_confident_learning(df, epochs=30, Ns_fraction=0.6, nu=1e-2)
    return df_cl


def apply_iflipper_correction(df):
    """Apply iFlipper fairness correction"""
    return apply_iflipper_fairness(df, error_fraction=0.1)


def calculate_fairness_metrics(df, label_col='best_controller_label'):
    """
    Calculate comprehensive fairness metrics based on:
    Farayola et al. (2026): Fairness-focused approach to recidivism prediction
    
    Metrics:
    1. Statistical Parity Difference (SPD) - Target: 0
    2. Disparate Impact (DI) - Target: 1 (valid range: [0.8, 1.2])
    3. Equal Opportunity Difference (EOD) - Target: 0
    4. Individual Fairness (from iFlipper) - Consistency across similar instances
    """
    logger.info("\n" + "=" * 70)
    logger.info("FAIRNESS METRICS (Farayola et al., 2026)")
    logger.info("=" * 70)
    
    # Define privileged and disadvantaged groups
    privileged = df[df[FAIRNESS_CONFIG['sensitive_attr']].isin(
        FAIRNESS_CONFIG['privileged_groups']
    )].copy()
    disadvantaged = df[df[FAIRNESS_CONFIG['sensitive_attr']].isin(
        FAIRNESS_CONFIG['disadvantaged_groups']
    )].copy()
    
    n_privileged = len(privileged)
    n_disadvantaged = len(disadvantaged)
    
    # Create binary label for favorable outcome (best controller selected)
    # For simplicity, we use the AHP score threshold
    if 'ahp_score' in df.columns:
        ahp_median = df['ahp_score'].median()
        privileged['favorable'] = (privileged['ahp_score'] >= ahp_median).astype(int)
        disadvantaged['favorable'] = (disadvantaged['ahp_score'] >= ahp_median).astype(int)
    else:
        logger.warning("ahp_score column not found, using ground truth labels")
        # Fallback: use label encoding
        le = LabelEncoder()
        le.fit(df[label_col].unique())
        privileged['favorable'] = (le.transform(privileged[label_col]) > 0).astype(int)
        disadvantaged['favorable'] = (le.transform(disadvantaged[label_col]) > 0).astype(int)
    
    # ========================================================================
    # 1. STATISTICAL PARITY DIFFERENCE (SPD)
    # ========================================================================
    # SPD = P(favorable | privileged) - P(favorable | disadvantaged)
    # Target: 0 (both groups have equal probability of favorable outcome)
    
    if n_privileged > 0:
        p_favorable_privileged = privileged['favorable'].sum() / n_privileged
    else:
        p_favorable_privileged = 0
    
    if n_disadvantaged > 0:
        p_favorable_disadvantaged = disadvantaged['favorable'].sum() / n_disadvantaged
    else:
        p_favorable_disadvantaged = 0
    
    spd = p_favorable_privileged - p_favorable_disadvantaged
    
    # ========================================================================
    # 2. DISPARATE IMPACT (DI)
    # ========================================================================
    # DI = P(favorable | disadvantaged) / P(favorable | privileged)
    # Target: 1 (both groups have equal selection rates)
    # Valid range: [0.8, 1.2] (80% rule in legal context)
    
    if p_favorable_privileged > 0:
        di = p_favorable_disadvantaged / p_favorable_privileged
    else:
        di = 0 if p_favorable_disadvantaged == 0 else float('inf')
    
    # ========================================================================
    # 3. EQUAL OPPORTUNITY DIFFERENCE (EOD)
    # ========================================================================
    # EOD = P(pred=1 | actual=1, privileged) - P(pred=1 | actual=1, disadvantaged)
    # Target: 0 (equal true positive rates across groups)
    # Requires true labels
    
    if 'best_controller_label' in df.columns:
        # Create binary true label (positive class = top controller)
        le = LabelEncoder()
        le.fit(df[label_col].unique())
        df_temp = df.copy()
        df_temp['y_true'] = le.transform(df_temp[label_col])
        y_true_binary = (df_temp['y_true'] > 0).astype(int)  # top controller = 1
        
        privileged['y_true'] = y_true_binary[privileged.index]
        disadvantaged['y_true'] = y_true_binary[disadvantaged.index]
        
        # True Positive Rate (TPR) per group
        priv_positives = (privileged['y_true'] == 1).sum()
        dis_positives = (disadvantaged['y_true'] == 1).sum()
        
        if priv_positives > 0:
            tpr_privileged = (
                ((privileged['y_true'] == 1) & (privileged['favorable'] == 1)).sum() / priv_positives
            )
        else:
            tpr_privileged = 0
        
        if dis_positives > 0:
            tpr_disadvantaged = (
                ((disadvantaged['y_true'] == 1) & (disadvantaged['favorable'] == 1)).sum() / dis_positives
            )
        else:
            tpr_disadvantaged = 0
        
        eod = tpr_privileged - tpr_disadvantaged
    else:
        eod = 0
        tpr_privileged = 0
        tpr_disadvantaged = 0
    
    # ========================================================================
    # 4. INDIVIDUAL FAIRNESS
    # ========================================================================
    # Measure consistency: similar instances should receive similar decisions
    # Based on feature similarity (iFlipper principle)
    
    feature_cols = [col for col in df.columns if col in [
        'throughput_mbps', 'latency_ms', 'packet_loss_percent',
        'jitter_ms', 'cpu_usage_percent', 'memory_usage_mb',
        'flow_setup_time_ms', 'nodes'
    ]]
    
    if len(feature_cols) > 0:
        X = df[feature_cols].fillna(0).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Compute pairwise similarity (Euclidean distance)
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(X_scaled)
        
        # Define similar pairs: distance < 25th percentile
        distance_threshold = np.percentile(distances[distances > 0], 25)
        
        # For each pair of similar instances, check if they have same favorable outcome
        le_check = LabelEncoder()
        le_check.fit(df[label_col].unique())
        predictions = le_check.transform(df[label_col])
        pred_binary = (predictions > 0).astype(int)
        
        inconsistencies = 0
        similar_pairs = 0
        
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                if 0 < distances[i, j] <= distance_threshold:
                    similar_pairs += 1
                    if pred_binary[i] != pred_binary[j]:
                        inconsistencies += 1
        
        if similar_pairs > 0:
            individual_fairness = 1.0 - (inconsistencies / similar_pairs)
        else:
            individual_fairness = 1.0
    else:
        individual_fairness = 1.0
    
    # ========================================================================
    # COMPILE RESULTS
    # ========================================================================
    metrics = {
        # Demographic information
        'n_privileged': n_privileged,
        'n_disadvantaged': n_disadvantaged,
        'p_favorable_privileged': round(p_favorable_privileged, 4),
        'p_favorable_disadvantaged': round(p_favorable_disadvantaged, 4),
        
        # Fairness metrics
        'spd': round(spd, 4),  # Target: 0
        'di': round(di, 4),    # Target: 1, valid: [0.8, 1.2]
        'eod': round(eod, 4),  # Target: 0
        'tpr_privileged': round(tpr_privileged, 4),
        'tpr_disadvantaged': round(tpr_disadvantaged, 4),
        'individual_fairness': round(individual_fairness, 4),  # Target: 1.0
        
        # Fair thresholds (from paper)
        'spd_fair': abs(spd) < 0.1,  # SPD < 0.1
        'di_fair': 0.8 <= di <= 1.2,  # DI in [0.8, 1.2]
        'eod_fair': abs(eod) < 0.1,  # EOD < 0.1
        'individual_fair': individual_fairness > 0.9,  # IF > 0.9
    }
    
    # ========================================================================
    # LOGGING
    # ========================================================================
    logger.info(f"\n[GROUP DISTRIBUTION]")
    logger.info(f"  Privileged groups:     {n_privileged:4d} samples ({100*n_privileged/(n_privileged+n_disadvantaged):.1f}%)")
    logger.info(f"  Disadvantaged groups:  {n_disadvantaged:4d} samples ({100*n_disadvantaged/(n_privileged+n_disadvantaged):.1f}%)")
    
    logger.info(f"\n[1. STATISTICAL PARITY DIFFERENCE (SPD)]")
    logger.info(f"  Formula: P(favorable|privileged) - P(favorable|disadvantaged)")
    logger.info(f"  P(favorable|privileged):     {p_favorable_privileged:.4f}")
    logger.info(f"  P(favorable|disadvantaged):  {p_favorable_disadvantaged:.4f}")
    logger.info(f"  SPD: {spd:.4f}")
    logger.info(f"  Target: 0 | Fair: {metrics['spd_fair']} (|SPD| < 0.1)")
    
    logger.info(f"\n[2. DISPARATE IMPACT (DI)]")
    logger.info(f"  Formula: P(favorable|disadvantaged) / P(favorable|privileged)")
    logger.info(f"  DI: {di:.4f}")
    logger.info(f"  Target: 1 | Valid range: [0.8, 1.2]")
    logger.info(f"  Fair: {metrics['di_fair']}")
    
    logger.info(f"\n[3. EQUAL OPPORTUNITY DIFFERENCE (EOD)]")
    logger.info(f"  Formula: TPR(privileged) - TPR(disadvantaged)")
    logger.info(f"  TPR(privileged):     {tpr_privileged:.4f}")
    logger.info(f"  TPR(disadvantaged):  {tpr_disadvantaged:.4f}")
    logger.info(f"  EOD: {eod:.4f}")
    logger.info(f"  Target: 0 | Fair: {metrics['eod_fair']} (|EOD| < 0.1)")
    
    logger.info(f"\n[4. INDIVIDUAL FAIRNESS]")
    logger.info(f"  Consistency across {similar_pairs if len(feature_cols) > 0 else 0} similar instance pairs")
    logger.info(f"  Individual Fairness: {individual_fairness:.4f}")
    logger.info(f"  Target: 1.0 | Fair: {metrics['individual_fair']} (IF > 0.9)")
    
    logger.info(f"\n[OVERALL FAIRNESS STATUS]")
    fair_count = sum([
        metrics['spd_fair'],
        metrics['di_fair'],
        metrics['eod_fair'],
        metrics['individual_fair']
    ])
    logger.info(f"  Metrics achieved: {fair_count}/4")
    logger.info("=" * 70)
    
    return metrics


def generate_fairness_report(metrics_before, metrics_after_cl, metrics_after_iflipper):
    """
    Generate comprehensive fairness report comparing all stages
    """
    logger.info("\n" + "=" * 70)
    logger.info("FAIRNESS IMPROVEMENT REPORT")
    logger.info("=" * 70)
    
    stages = {
        'Original (AHP)': metrics_before,
        'After CL': metrics_after_cl,
        'After iFlipper': metrics_after_iflipper
    }
    
    # Create comparison table
    logger.info("\n[METRIC COMPARISON ACROSS STAGES]")
    logger.info(f"{'Metric':<30} {'Before':<15} {'After CL':<15} {'After iFlipper':<15}")
    logger.info("-" * 75)
    
    # SPD
    spd_before = stages['Original (AHP)']['spd']
    spd_cl = stages['After CL']['spd']
    spd_if = stages['After iFlipper']['spd']
    logger.info(f"{'SPD (Target: 0)':<30} {spd_before:>6.4f}         {spd_cl:>6.4f}         {spd_if:>6.4f}")
    logger.info(f"{'  Fair? (|SPD| < 0.1)':<30} {str(stages['Original (AHP)']['spd_fair']):>6}         {str(stages['After CL']['spd_fair']):>6}         {str(stages['After iFlipper']['spd_fair']):>6}")
    
    # DI
    di_before = stages['Original (AHP)']['di']
    di_cl = stages['After CL']['di']
    di_if = stages['After iFlipper']['di']
    logger.info(f"\n{'DI (Target: 1, [0.8-1.2])':<30} {di_before:>6.4f}         {di_cl:>6.4f}         {di_if:>6.4f}")
    logger.info(f"{'  Fair? (0.8 ≤ DI ≤ 1.2)':<30} {str(stages['Original (AHP)']['di_fair']):>6}         {str(stages['After CL']['di_fair']):>6}         {str(stages['After iFlipper']['di_fair']):>6}")
    
    # EOD
    eod_before = stages['Original (AHP)']['eod']
    eod_cl = stages['After CL']['eod']
    eod_if = stages['After iFlipper']['eod']
    logger.info(f"\n{'EOD (Target: 0)':<30} {eod_before:>6.4f}         {eod_cl:>6.4f}         {eod_if:>6.4f}")
    logger.info(f"{'  Fair? (|EOD| < 0.1)':<30} {str(stages['Original (AHP)']['eod_fair']):>6}         {str(stages['After CL']['eod_fair']):>6}         {str(stages['After iFlipper']['eod_fair']):>6}")
    
    # Individual Fairness
    if_before = stages['Original (AHP)']['individual_fairness']
    if_cl = stages['After CL']['individual_fairness']
    if_if = stages['After iFlipper']['individual_fairness']
    logger.info(f"\n{'IF (Target: 1.0)':<30} {if_before:>6.4f}         {if_cl:>6.4f}         {if_if:>6.4f}")
    logger.info(f"{'  Fair? (IF > 0.9)':<30} {str(stages['Original (AHP)']['individual_fair']):>6}         {str(stages['After CL']['individual_fair']):>6}         {str(stages['After iFlipper']['individual_fair']):>6}")
    
    # Overall fairness achievement
    logger.info(f"\n[OVERALL FAIRNESS STATUS]")
    before_fair = sum([
        stages['Original (AHP)']['spd_fair'],
        stages['Original (AHP)']['di_fair'],
        stages['Original (AHP)']['eod_fair'],
        stages['Original (AHP)']['individual_fair']
    ])
    after_cl_fair = sum([
        stages['After CL']['spd_fair'],
        stages['After CL']['di_fair'],
        stages['After CL']['eod_fair'],
        stages['After CL']['individual_fair']
    ])
    after_if_fair = sum([
        stages['After iFlipper']['spd_fair'],
        stages['After iFlipper']['di_fair'],
        stages['After iFlipper']['eod_fair'],
        stages['After iFlipper']['individual_fair']
    ])
    
    logger.info(f"  Original:      {before_fair}/4 metrics achieved")
    logger.info(f"  After CL:      {after_cl_fair}/4 metrics achieved")
    logger.info(f"  After iFlipper: {after_if_fair}/4 metrics achieved")
    
    logger.info("=" * 70)
    
    return {
        'before': before_fair,
        'after_cl': after_cl_fair,
        'after_iflipper': after_if_fair
    }



    logger.info("=" * 60)
    logger.info("FAIRNESS PIPELINE: CL + iFlipper")
    logger.info("Based on CL.txt and iFlipper_Demo.ipynb")
    logger.info("=" * 60)
    
    df = load_dataset(DATASET_A)
    if df is None:
        logger.error("Please run codeICCRAIDS.py first")
        return
    
    logger.info(f"Original dataset: {len(df)} samples")
    
    metrics_before = calculate_fairness_metrics(df)
    
    logger.info("\n" + "=" * 60)
    logger.info("Stage 1: CONFIDENT LEARNING")
    logger.info("=" * 60)
    df_cl = apply_confident_learning_filter(df)
    
    df_clean = df_cl[~df_cl.get('is_noisy_cl', pd.Series([False]*len(df_cl)))].copy()
    if 'is_noisy_cl' in df_clean.columns:
        df_clean = df_clean.drop(columns=['is_noisy_cl'])
    
    noisy_removed = len(df) - len(df_clean)
    logger.info(f"Removed {noisy_removed} noisy samples, kept {len(df_clean)}")
    
    df_clean.to_csv(DATASET_A_CLEAN, index=False)
    logger.info(f"Clean dataset: {DATASET_A_CLEAN}")
    
    metrics_cl = calculate_fairness_metrics(df_clean)
    
    logger.info("\n" + "=" * 60)
    logger.info("Stage 2: IFLIPPER")
    logger.info("=" * 60)
    df_fair = apply_iflipper_correction(df_clean)
    
    df_fair.to_csv(DATASET_B, index=False)
    logger.info(f"Fair dataset: {DATASET_B}")
    
    metrics_after = calculate_fairness_metrics(df_fair)
    
    pipeline_results = {
        'original_samples': len(df),
        'clean_samples': len(df_clean),
        'fair_samples': len(df_fair),
        'noise_removed': noisy_removed,
        'metrics_before': metrics_before,
        'metrics_after_cl': metrics_cl,
        'metrics_after_iflipper': metrics_after
    }
    
    results_file = os.path.join(OUTPUT_DIR, 'pipeline_results.json')
    with open(results_file, 'w') as f:
        json.dump(pipeline_results, f, indent=2)
    logger.info(f"\nResults saved: {results_file}")
    
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Original (AHP): {pipeline_results['original_samples']}")
    logger.info(f"After CL:        {pipeline_results['clean_samples']}")
    logger.info(f"After iFlipper: {pipeline_results['fair_samples']}")
    logger.info(f"Fairness improvement:")
    logger.info(f"  Gap: {metrics_before['score_gap']:.4f} -> {metrics_after['score_gap']:.4f}")
    logger.info(f"  SPD: {metrics_before['statistical_parity_difference']:.4f} -> {metrics_after['statistical_parity_difference']:.4f}")
    
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