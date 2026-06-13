#!/usr/bin/env python3
import csv, os, sys, json, logging, warnings
import numpy as np
import pandas as pd
from collections import defaultdict

warnings.filterwarnings('ignore')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, will generate text-based charts")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASET_A = os.path.join(DATA_DIR, 'sdn_dataset_ahp.csv')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

COLORS = {
    'ryu': '#e74c3c',
    'onos': '#3498db', 
    'floodlight': '#2ecc71'
}


def load_dataset(filepath):
    if not os.path.exists(filepath):
        logger.error(f"Dataset not found: {filepath}")
        return None
    return pd.read_csv(filepath)


def plot_performance_by_controller(df, output_dir):
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Skipping controller performance plot")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('SDN Controller Performance Comparison', fontsize=14, fontweight='bold')
    
    metrics = ['throughput_mbps', 'latency_ms', 'packet_loss_percent', 
               'jitter_ms', 'cpu_usage_percent', 'memory_usage_mb']
    titles = ['Throughput (Mbps)', 'Latency (ms)', 'Packet Loss (%)', 
             'Jitter (ms)', 'CPU Usage (%)', 'Memory (MB)']
    
    controllers = df['controller_name'].unique()
    x = np.arange(len(controllers))
    width = 0.6
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 3, idx % 3]
        means = [df[df['controller_name'] == c][metric].mean() for c in controllers]
        stds = [df[df['controller_name'] == c][metric].std() for c in controllers]
        
        bars = ax.bar(x, means, width, yerr=stds, capsize=5, 
                    color=[COLORS.get(c, '#95a5a6') for c in controllers],
                    edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Controller')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([c.upper() for c in controllers])
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'controller_performance.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_ahp_scores_by_topology(df, output_dir):
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Skipping AHP scores plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('AHP Score Analysis by Topology', fontsize=14, fontweight='bold')
    
    topo_order = ['star', 'tree', 'linear', 'mesh', 'ring']
    controllers = df['controller_name'].unique()
    
    ax1 = axes[0]
    topo_means = []
    for topo in topo_order:
        subset = df[df['topology_type'] == topo]
        if len(subset) > 0:
            means = [subset[subset['controller_name'] == c]['ahp_score'].mean() for c in controllers]
            topo_means.append(means)
    
    x = np.arange(len(topo_order))
    width = 0.25
    
    for i, ctrl in enumerate(controllers):
        values = [topo_means[j][i] if j < len(topo_means) else 0 for j in range(len(topo_order))]
        ax1.bar(x + i * width, values, width, label=ctrl.upper(), 
                color=COLORS.get(ctrl, '#95a5a6'))
    
    ax1.set_xlabel('Topology Type')
    ax1.set_ylabel('AHP Score')
    ax1.set_title('Average AHP Score by Topology')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([t.upper() for t in topo_order])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    ax2 = axes[1]
    privileged = df[df['topology_type'].isin(['star', 'tree', 'linear'])]
    disadvantaged = df[df['topology_type'].isin(['mesh', 'ring'])]
    
    priv_scores = privileged.groupby('controller_name')['ahp_score'].mean()
    dis_scores = disadvantaged.groupby('controller_name')['ahp_score'].mean()
    
    x = np.arange(len(controllers))
    width = 0.35
    
    ax2.bar(x - width/2, [priv_scores.get(c, 0) for c in controllers], 
            width, label='Privileged', color='#9b59b6', alpha=0.8)
    ax2.bar(x + width/2, [dis_scores.get(c, 0) for c in controllers], 
            width, label='Disadvantaged', color='#e67e22', alpha=0.8)
    
    ax2.set_xlabel('Controller')
    ax2.set_ylabel('AHP Score')
    ax2.set_title('Privileged vs Disadvantaged Groups')
    ax2.set_xticks(x)
    ax2.set_xticklabels([c.upper() for c in controllers])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'ahp_scores_topology.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_fairness_gap(df, output_dir):
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Skipping fairness gap plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    privileged_topo = ['star', 'tree', 'linear']
    disadvantaged_topo = ['mesh', 'ring']
    
    controllers = df['controller_name'].unique()
    
    gap_data = []
    for ctrl in controllers:
        subset = df[df['controller_name'] == ctrl]
        priv = subset[subset['topology_type'].isin(privileged_topo)]['ahp_score'].mean()
        dis = subset[subset['topology_type'].isin(disadvantaged_topo)]['ahp_score'].mean()
        gap = priv - dis
        gap_data.append({'controller': ctrl.upper(), 'gap': gap})
    
    colors = [COLORS.get(c, '#95a5a6') for c in controllers]
    bars = ax.bar([d['controller'] for d in gap_data], [d['gap'] for d in gap_data], 
                 color=colors, edgecolor='black', linewidth=0.5)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=0.1, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Fairness Threshold')
    ax.axhline(y=-0.1, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Controller')
    ax.set_ylabel('AHP Score Gap')
    ax.set_title('Fairness Gap: Privileged - Disadvantaged')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'fairness_gap.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_iteration_stability(df, output_dir):
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Skipping iteration stability plot")
        return
    
    if 'iteration' not in df.columns or df['iteration'].isna().all():
        logger.info("No iteration data for stability analysis")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for ctrl in df['controller_name'].unique():
        subset = df[df['controller_name'] == ctrl]
        iterations = sorted(subset['iteration'].unique())
        means = [subset[subset['iteration'] == i]['ahp_score'].mean() for i in iterations]
        stds = [subset[subset['iteration'] == i]['ahp_score'].std() for i in iterations]
        
        ax.errorbar(iterations, means, yerr=stds, marker='o', 
                   label=ctrl.upper(), color=COLORS.get(ctrl, '#95a5a6'),
                   capsize=5, linewidth=2)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('AHP Score')
    ax.set_title('Iteration Stability Analysis')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'iteration_stability.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_scenario_heatmap(df, output_dir):
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Skipping heatmap")
        return
    
    scenario_pivot = df.pivot_table(
        values='ahp_score', 
        index='scenario_name', 
        columns='controller_name',
        aggfunc='mean'
    )
    
    if len(scenario_pivot) > 0:
        fig, ax = plt.subplots(figsize=(12, max(8, len(scenario_pivot) * 0.4)))
        
        sns.heatmap(scenario_pivot, annot=True, fmt='.3f', cmap='YlOrRd',
                    ax=ax, cbar_kws={'label': 'AHP Score'})
        
        ax.set_title('AHP Score Heatmap: Scenario vs Controller')
        ax.set_xlabel('Controller')
        ax.set_ylabel('Scenario')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'scenario_heatmap.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {output_path}")


def plot_dataset_distribution(df, output_dir):
    if not MATPLOTLIB_AVAILABLE:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1 = axes[0]
    topo_counts = df['topology_type'].value_counts()
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    ax1.pie(topo_counts, labels=topo_counts.index.str.upper(), autopct='%1.1f%%',
            colors=colors[:len(topo_counts)])
    ax1.set_title('Topology Distribution')
    
    ax2 = axes[1]
    ctrl_counts = df['controller_name'].value_counts()
    ctrl_colors = [COLORS.get(c, '#95a5a6') for c in ctrl_counts.index]
    ax2.pie(ctrl_counts, labels=ctrl_counts.index.str.upper(), autopct='%1.1f%%',
            colors=ctrl_colors)
    ax2.set_title('Controller Distribution')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'dataset_distribution.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def generate_text_charts(df, output_dir):
    logger.info("\n--- Generating Text-Based Summary ---")
    
    report = []
    report.append("=" * 60)
    report.append("DATASET SUMMARY (Text-Based)")
    report.append("=" * 60)
    
    report.append(f"\nTotal Samples: {len(df)}")
    report.append(f"Controllers: {', '.join(df['controller_name'].unique())}")
    report.append(f"Topologies: {', '.join(df['topology_type'].unique())}")
    
    report.append("\n--- Performance by Controller ---")
    for ctrl in df['controller_name'].unique():
        subset = df[df['controller_name'] == ctrl]
        report.append(f"\n{ctrl.upper()}:")
        report.append(f"  Avg Throughput: {subset['throughput_mbps'].mean():.2f} Mbps")
        report.append(f"  Avg Latency: {subset['latency_ms'].mean():.2f} ms")
        report.append(f"  Avg CPU: {subset['cpu_usage_percent'].mean():.2f}%")
        report.append(f"  Avg AHP Score: {subset['ahp_score'].mean():.4f}")
    
    report.append("\n--- Best Controller by Topology ---")
    for topo in df['topology_type'].unique():
        subset = df[df['topology_type'] == topo]
        best = subset.groupby('controller_name')['ahp_score'].mean().idxmax()
        score = subset.groupby('controller_name')['ahp_score'].mean().max()
        report.append(f"  {topo.upper()}: {best.upper()} ({score:.4f})")
    
    report.append("\n--- Privileged vs Disadvantaged ---")
    privileged = df[df['topology_type'].isin(['star', 'tree', 'linear'])]
    disadvantaged = df[df['topology_type'].isin(['mesh', 'ring'])]
    report.append(f"  Privileged (Star/Tree/Linear): {privileged['ahp_score'].mean():.4f}")
    report.append(f"  Disadvantaged (Mesh/Ring): {disadvantaged['ahp_score'].mean():.4f}")
    report.append(f"  Gap: {privileged['ahp_score'].mean() - disadvantaged['ahp_score'].mean():.4f}")
    
    text = "\n".join(report)
    print(text)
    
    report_file = os.path.join(output_dir, 'dataset_summary.txt')
    with open(report_file, 'w') as f:
        f.write(text)
    logger.info(f"Saved: {report_file}")


def main():
    logger.info("=" * 60)
    logger.info("VISUALIZATION GENERATION")
    logger.info("=" * 60)
    
    df = load_dataset(DATASET_A)
    if df is None:
        logger.error("Please run codeICCRAIDS.py first to generate data")
        return
    
    logger.info(f"Loaded {len(df)} samples")
    
    if MATPLOTLIB_AVAILABLE:
        plot_performance_by_controller(df, OUTPUT_DIR)
        plot_ahp_scores_by_topology(df, OUTPUT_DIR)
        plot_fairness_gap(df, OUTPUT_DIR)
        plot_iteration_stability(df, OUTPUT_DIR)
        plot_scenario_heatmap(df, OUTPUT_DIR)
        plot_dataset_distribution(df, OUTPUT_DIR)
    else:
        logger.warning("matplotlib not available, using text-only output")
    
    generate_text_charts(df, OUTPUT_DIR)
    
    logger.info(f"\nAll outputs saved to: {OUTPUT_DIR}")
    logger.info("Visualization complete")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)