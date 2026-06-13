#!/usr/bin/env python3
import csv, os, sys, json, logging, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import traceback

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASET_A = os.path.join(DATA_DIR, 'sdn_dataset_ahp.csv')
DATASET_B = os.path.join(DATA_DIR, 'sdn_dataset_fair.csv')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

FAIRNESS_METRICS = {
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


def calculate_statistical_parity_difference(df, target_col, sensitive_col):
    privileged_mask = df[sensitive_col].isin(FAIRNESS_METRICS['privileged_groups'])
    disadvantaged_mask = df[sensitive_col].isin(FAIRNESS_METRICS['disadvantaged_groups'])
    
    privileged_positive = (df[privileged_mask][target_col] == 1).mean()
    disadvantaged_positive = (df[disadvantaged_mask][target_col] == 1).mean()
    
    return privileged_positive - disadvantaged_positive


def calculate_equal_opportunity_difference(df, target_col, sensitive_col, true_label_col):
    privileged_mask = df[sensitive_col].isin(FAIRNESS_METRICS['privileged_groups'])
    disadvantaged_mask = df[sensitive_col].isin(FAIRNESS_METRICS['disadvantaged_groups'])
    
    priv_true = df[privileged_mask][df[true_label_col] == 1]
    dis_true = df[disadvantaged_mask][df[true_label_col] == 1]
    
    priv_tpr = (priv_true[target_col] == 1).mean() if len(priv_true) > 0 else 0
    dis_tpr = (dis_true[target_col] == 1).mean() if len(dis_true) > 0 else 0
    
    return priv_tpr - dis_tpr


def calculate_fairness_metrics(df, label_col):
    results = {}
    sensitive_col = FAIRNESS_METRICS['sensitive_attr']
    
    results['statistical_parity'] = calculate_statistical_parity_difference(
        df, label_col, sensitive_col
    )
    results['equal_opportunity'] = calculate_equal_opportunity_difference(
        df, label_col, sensitive_col, label_col.replace('pred', 'true')
    )
    
    privileged_mask = df[sensitive_col].isin(FAIRNESS_METRICS['privileged_groups'])
    disadvantaged_mask = df[sensitive_col].isin(FAIRNESS_METRICS['disadvantaged_groups'])
    
    results['privileged_accuracy'] = accuracy_score(
        df[privileged_mask][label_col.replace('pred', 'label')],
        df[privileged_mask][label_col]
    )
    results['disadvantaged_accuracy'] = accuracy_score(
        df[disadvantaged_mask][label_col.replace('pred', 'label')],
        df[disadvantaged_mask][label_col]
    )
    results['accuracy_gap'] = results['privileged_accuracy'] - results['disadvantaged_accuracy']
    
    return results


def evaluate_model(df, dataset_name):
    feature_cols = ['throughput_mbps', 'latency_ms', 'packet_loss_percent', 
                   'jitter_ms', 'cpu_usage_percent', 'memory_usage_mb',
                   'flow_setup_time_ms', 'nodes']
    
    if 'topology_type' not in df.columns:
        logger.error("topology_type column missing")
        return None
    
    df_features = df[feature_cols].copy()
    df_features = df_features.fillna(0)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(df_features)
    
    label_encoders = {}
    y = None
    if 'best_controller_label' in df.columns:
        le = LabelEncoder()
        y = le.fit_transform(df['best_controller_label'])
        label_encoders['label'] = le
    else:
        logger.warning("No label column found, using is_ryu as default")
        y = df.get('is_ryu', 0).values
    
    results = {
        'dataset': dataset_name,
        'n_samples': len(df),
        'n_features': len(feature_cols)
    }
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    dt_model = DecisionTreeClassifier(random_state=42)
    
    try:
        rf_scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')
        dt_scores = cross_val_score(dt_model, X, y, cv=5, scoring='accuracy')
        
        results['rf_accuracy_mean'] = round(rf_scores.mean(), 4)
        results['rf_accuracy_std'] = round(rf_scores.std(), 4)
        results['dt_accuracy_mean'] = round(dt_scores.mean(), 4)
        results['dt_accuracy_std'] = round(dt_scores.std(), 4)
        
        logger.info(f"{dataset_name} - RF Accuracy: {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")
        logger.info(f"{dataset_name} - DT Accuracy: {dt_scores.mean():.4f} ± {dt_scores.std():.4f}")
        
    except Exception as e:
        logger.error(f"Model evaluation error: {e}")
        results['error'] = str(e)
    
    return results


def compare_datasets():
    logger.info("=" * 60)
    logger.info("DATASET EVALUATION: Comparing Dataset A vs Dataset B")
    logger.info("=" * 60)
    
    df_a = load_dataset(DATASET_A)
    df_b = load_dataset(DATASET_B)
    
    if df_a is None:
        logger.error("Dataset A not found. Run codeICCRAIDS.py first.")
        return
    
    eval_results = []
    
    logger.info("\n--- Dataset A (AHP Labels) ---")
    results_a = evaluate_model(df_a, "Dataset A")
    if results_a:
        eval_results.append(results_a)
    
    if df_b is not None:
        logger.info("\n--- Dataset B (Fair Labels) ---")
        results_b = evaluate_model(df_b, "Dataset B")
        if results_b:
            eval_results.append(results_b)
        
        logger.info("\n--- Fairness Comparison ---")
        fairness_a = calculate_fairness_metrics(df_a, 'best_controller_label')
        fairness_b = calculate_fairness_metrics(df_b, 'best_controller_label')

        
        logger.info(f"Dataset A - Statistical Parity: {fairness_a['statistical_parity']:.4f}")
        logger.info(f"Dataset B - Statistical Parity: {fairness_b['statistical_parity']:.4f}")
        logger.info(f"Dataset A - Accuracy Gap: {fairness_a['accuracy_gap']:.4f}")
        logger.info(f"Dataset B - Accuracy Gap: {fairness_b['accuracy_gap']:.4f}")
        
        improvement_spd = abs(fairness_a['statistical_parity']) - abs(fairness_b['statistical_parity'])
        improvement_gap = abs(fairness_a['accuracy_gap']) - abs(fairness_b['accuracy_gap'])
        
        logger.info(f"\n--- Fairness Improvement ---")
        logger.info(f"Statistical Parity Improvement: {improvement_spd:+.4f}")
        logger.info(f"Accuracy Gap Improvement: {improvement_gap:+.4f}")
        
        eval_results[0].update(fairness_a)
        eval_results[1].update(fairness_b)
    else:
        logger.warning("Dataset B not found - skipping comparison")
    
    results_file = os.path.join(OUTPUT_DIR, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(eval_results, f, indent=2)
    logger.info(f"\nResults saved to {results_file}")
    
    return eval_results


def generate_summary_report():
    results_file = os.path.join(OUTPUT_DIR, 'evaluation_results.json')
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
        report.append(f"  Samples: {res.get('n_samples', 'N/A')}")
        report.append(f"  Features: {res.get('n_features', 'N/A')}")
        
        if 'rf_accuracy_mean' in res:
            report.append(f"  Random Forest Accuracy: {res['rf_accuracy_mean']:.4f} ± {res['rf_accuracy_std']:.4f}")
        if 'dt_accuracy_mean' in res:
            report.append(f"  Decision Tree Accuracy: {res['dt_accuracy_mean']:.4f} ± {res['dt_accuracy_std']:.4f}")
        
        if 'statistical_parity' in res:
            report.append(f"  Statistical Parity Diff: {res['statistical_parity']:.4f}")
        if 'accuracy_gap' in res:
            report.append(f"  Accuracy Gap: {res['accuracy_gap']:.4f}")
    
    report_text = "\n".join(report)
    print(report_text)
    
    report_file = os.path.join(OUTPUT_DIR, 'evaluation_report.txt')
    with open(report_file, 'w') as f:
        f.write(report_text)
    logger.info(f"Report saved to {report_file}")


def main():
    logger.info("Starting evaluation...")
    comparison_results = compare_datasets()
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
