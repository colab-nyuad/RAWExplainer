"""
Visualize and Compare Explanation Subgraph Structures

This script generates visualizations comparing the structural properties of explanatory 
subgraphs from multiple explanation files, focusing on valid (non-padding) edges.

Features:
- Distribution plots for key metrics (density, clustering, connectivity)
- Side-by-side comparisons across models
- Sample subgraph visualizations
- Aggregate statistics plots

Usage:
    python visualize_subgraph_structure.py --files FILE1 FILE2 [FILE3 ...] --labels LABEL1 LABEL2 [LABEL3 ...]
    
Example:
    python visualize_subgraph_structure.py \
        --files model1/explanations.pt model2/explanations.pt \
        --labels "PGExplainer" "RAWExplainer" \
        --output-dir ./plots/expl_subgraph_comp
"""

import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import networkx as nx
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")


def load_kg_data(dataset_path):
    """Load the knowledge graph dataset to map edge IDs to actual triples"""
    print(f"\nLoading knowledge graph from: {dataset_path}")
    
    processed_file = Path(dataset_path) / 'processed' / 'data.pt'
    
    if not processed_file.exists():
        print(f"ERROR: Dataset file not found: {processed_file}")
        return None
    
    data_list = torch.load(processed_file, map_location='cpu')
    train_data = data_list[0]
    
    num_relations = train_data.edge_type.max().item() + 1
    
    print(f"Knowledge Graph loaded:")
    print(f"  Num nodes: {train_data.num_nodes}")
    print(f"  Num edges: {train_data.edge_index.shape[1]}")
    print(f"  Num relations: {num_relations}")
    
    return train_data


def get_valid_edges(explanation_row):
    """Extract valid (non -1) edge IDs from an explanation row"""
    query = explanation_row[:2]
    edge_ids = explanation_row[2:]
    valid_edge_ids = edge_ids[edge_ids != -1]
    return query, valid_edge_ids


def build_subgraph_from_edges(edge_ids, kg_data):
    """Build a NetworkX graph from edge IDs using the KG data"""
    G = nx.DiGraph()
    
    edge_ids_list = edge_ids.tolist() if isinstance(edge_ids, torch.Tensor) else edge_ids
    
    for edge_id in edge_ids_list:
        if edge_id < 0 or edge_id >= kg_data.edge_index.shape[1]:
            continue
        
        src = kg_data.edge_index[0, edge_id].item()
        tgt = kg_data.edge_index[1, edge_id].item()
        rel = kg_data.edge_type[edge_id].item()
        
        G.add_edge(src, tgt, relation=rel, edge_id=edge_id)
    
    return G


def compute_graph_metrics(G):
    """Compute structural metrics for a graph"""
    metrics = {}
    
    if G.number_of_nodes() == 0:
        return {'is_empty': True}
    
    metrics['num_nodes'] = G.number_of_nodes()
    metrics['num_edges'] = G.number_of_edges()
    metrics['is_empty'] = False
    
    # Density
    metrics['density'] = nx.density(G) if G.number_of_nodes() > 1 else 0
    
    # Connected components
    G_undirected = G.to_undirected()
    metrics['num_components'] = nx.number_connected_components(G_undirected)
    metrics['is_connected'] = nx.is_connected(G_undirected)
    
    # Largest component
    if metrics['num_components'] > 0:
        largest_cc = max(nx.connected_components(G_undirected), key=len)
        metrics['largest_component_size'] = len(largest_cc)
        metrics['largest_component_ratio'] = len(largest_cc) / G.number_of_nodes()
    else:
        metrics['largest_component_size'] = 0
        metrics['largest_component_ratio'] = 0
    
    # Clustering
    try:
        metrics['avg_clustering'] = nx.average_clustering(G_undirected)
    except:
        metrics['avg_clustering'] = 0
    
    # Diameter
    try:
        if metrics['is_connected']:
            metrics['diameter'] = nx.diameter(G_undirected)
        else:
            largest_cc_subgraph = G_undirected.subgraph(largest_cc)
            if largest_cc_subgraph.number_of_nodes() > 0:
                metrics['diameter'] = nx.diameter(largest_cc_subgraph)
            else:
                metrics['diameter'] = 0
    except:
        metrics['diameter'] = 0
    
    # Relations
    relations = [data['relation'] for _, _, data in G.edges(data=True)]
    metrics['num_unique_relations'] = len(set(relations))
    metrics['relation_diversity'] = len(set(relations)) / len(relations) if relations else 0
    
    return metrics


def analyze_explanation_file(filepath, kg_data, sample_size=None):
    """Analyze all explanations in a file and collect metrics"""
    print(f"\nAnalyzing: {filepath}")
    
    data = torch.load(filepath, map_location='cpu')
    
    num_samples = data.shape[0]
    if sample_size:
        num_samples = min(num_samples, sample_size)
    
    print(f"Total samples: {data.shape[0]}, analyzing: {num_samples}")
    
    all_metrics = []
    
    for i in range(num_samples):
        query, valid_edge_ids = get_valid_edges(data[i])
        
        if len(valid_edge_ids) == 0:
            continue
        
        G = build_subgraph_from_edges(valid_edge_ids, kg_data)
        metrics = compute_graph_metrics(G)
        
        if not metrics.get('is_empty', False):
            all_metrics.append(metrics)
    
    print(f"Collected {len(all_metrics)} valid subgraphs")
    return all_metrics


def create_distribution_comparison(all_results: Dict[str, List[Dict]], 
                                   metric_key: str, 
                                   title: str,
                                   xlabel: str,
                                   output_path: Path):
    """Create overlapping distribution plot for a specific metric"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for label, metrics_list in all_results.items():
        values = [m[metric_key] for m in metrics_list if metric_key in m]
        if values:
            ax.hist(values, bins=50, alpha=0.6, label=label, density=True)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def create_boxplot_comparison(all_results: Dict[str, List[Dict]], 
                              metric_key: str,
                              title: str,
                              ylabel: str,
                              output_path: Path):
    """Create box plot comparison for a specific metric"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data_to_plot = []
    labels = []
    
    for label, metrics_list in all_results.items():
        values = [m[metric_key] for m in metrics_list if metric_key in m]
        if values:
            data_to_plot.append(values)
            labels.append(label)
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                    showmeans=True, meanline=True)
    
    # Color the boxes
    colors = sns.color_palette("husl", len(data_to_plot))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=15, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def create_summary_comparison(all_results: Dict[str, List[Dict]], 
                              output_path: Path):
    """Create a comprehensive summary comparison plot"""
    metrics_to_plot = [
        ('num_nodes', 'Avg Nodes'),
        ('num_edges', 'Avg Edges'),
        ('density', 'Density'),
        ('num_components', 'Components'),
        ('avg_clustering', 'Clustering'),
        ('relation_diversity', 'Relation Diversity')
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (metric_key, metric_label) in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        labels = []
        means = []
        stds = []
        
        for label, metrics_list in all_results.items():
            values = [m[metric_key] for m in metrics_list if metric_key in m]
            if values:
                labels.append(label)
                means.append(np.mean(values))
                stds.append(np.std(values))
        
        x_pos = np.arange(len(labels))
        colors = sns.color_palette("husl", len(labels))
        
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color=colors)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.set_ylabel(metric_label, fontsize=10)
        ax.set_title(f'{metric_label}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean:.3f}',
                   ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Subgraph Structure Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def create_connectivity_comparison(all_results: Dict[str, List[Dict]], 
                                   output_path: Path):
    """Create connectivity statistics comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = []
    connected_pcts = []
    
    for label, metrics_list in all_results.items():
        connected_count = sum(1 for m in metrics_list if m.get('is_connected', False))
        total = len(metrics_list)
        pct = (connected_count / total * 100) if total > 0 else 0
        
        labels.append(label)
        connected_pcts.append(pct)
    
    x_pos = np.arange(len(labels))
    colors = sns.color_palette("husl", len(labels))
    
    bars = ax.bar(x_pos, connected_pcts, alpha=0.7, color=colors)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Connected Graphs Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels
    for bar, pct in zip(bars, connected_pcts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{pct:.1f}%',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def visualize_sample_subgraph(metrics_list: List[Dict], 
                              kg_data,
                              explanation_data,
                              label: str,
                              output_path: Path,
                              sample_idx: int = 0):
    """Visualize a sample subgraph from the explanations"""
    # Find a good representative sample (median number of nodes)
    node_counts = [m['num_nodes'] for m in metrics_list]
    median_nodes = np.median(node_counts)
    
    # Find sample closest to median
    best_idx = min(range(len(metrics_list)), 
                   key=lambda i: abs(metrics_list[i]['num_nodes'] - median_nodes))
    
    # Get the actual subgraph
    query, valid_edge_ids = get_valid_edges(explanation_data[best_idx])
    G = build_subgraph_from_edges(valid_edge_ids, kg_data)
    
    if G.number_of_nodes() == 0:
        print(f"  Empty graph for {label}, skipping visualization")
        return
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Use spring layout for better visualization
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue', 
                          alpha=0.9, ax=ax)
    
    # Draw edges with different colors for different relations
    relations = [data['relation'] for _, _, data in G.edges(data=True)]
    unique_relations = list(set(relations))
    color_map = {rel: plt.cm.tab20(i % 20) for i, rel in enumerate(unique_relations)}
    
    for u, v, data in G.edges(data=True):
        rel = data['relation']
        nx.draw_networkx_edges(G, pos, [(u, v)], edge_color=[color_map[rel]], 
                              alpha=0.6, width=2, arrows=True, 
                              arrowsize=15, arrowstyle='->', ax=ax)
    
    # Draw labels (only for graphs with few nodes)
    if G.number_of_nodes() <= 30:
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    
    metrics = metrics_list[best_idx]
    ax.set_title(f'Sample Subgraph - {label}\n'
                f'Nodes: {metrics["num_nodes"]}, Edges: {metrics["num_edges"]}, '
                f'Density: {metrics["density"]:.3f}, '
                f'Connected: {metrics["is_connected"]}',
                fontsize=12, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def create_scatter_comparison(all_results: Dict[str, List[Dict]], 
                              x_metric: str, y_metric: str,
                              xlabel: str, ylabel: str,
                              title: str,
                              output_path: Path):
    """Create scatter plot comparing two metrics"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = sns.color_palette("husl", len(all_results))
    
    for (label, metrics_list), color in zip(all_results.items(), colors):
        x_values = [m[x_metric] for m in metrics_list if x_metric in m and y_metric in m]
        y_values = [m[y_metric] for m in metrics_list if x_metric in m and y_metric in m]
        
        if x_values and y_values:
            ax.scatter(x_values, y_values, alpha=0.5, s=20, label=label, color=color)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    DEFAULT_FILES = [
        "/scratch/oge208/Graph-Transformer/NBFNet-PyG/explanation/PGExplainer/NBFNet/FB15k-237/finetune/test_explanations_factual_eval_hard_edge_mask_top_k_100.pt",
        "/scratch/oge208/Graph-Transformer/NBFNet-PyG/explanation/RAWExplainer/NBFNet/FB15k-237/2025-03-14-23-02-33-hyp-ola1_4/test_explanations_factual_eval_hard_edge_mask_top_k_100.pt",
        "/scratch/sl9030/gml/Graph-Transformer/NBFNet-PyG/explanation/RAWExplainer/NBFNet/FB15k-237/2026-02-02-17-00-13-var44-potential/test_explanations_factual_eval_hard_edge_mask_top_k_100.pt"
    ]
    DEFAULT_LABELS = ["PGExplainer", "hyp-ola1_4", "var44"]
    DEFAULT_DATASET = "/scratch/oge208/Graph-Transformer/NBFNet-PyG/datasets/knowledge_graphs/FB15k-237"

    DEFAULT_OUTPUT_DIR = f"./plots/expl_subgraph_comp_{100}k"
    DEFAULT_SAMPLE_SIZE = 5000
    
    parser = argparse.ArgumentParser(
        description='Visualize and compare subgraph structures in explanation files',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--files', nargs='+', default=DEFAULT_FILES,
                       help='Explanation files to compare')
    parser.add_argument('--labels', nargs='+', default=DEFAULT_LABELS,
                       help='Labels for each file')
    parser.add_argument('--dataset-path', default=DEFAULT_DATASET,
                       help='Path to dataset directory')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR,
                       help='Directory to save visualizations')
    parser.add_argument('--sample-size', type=int, default=DEFAULT_SAMPLE_SIZE,
                       help='Number of samples to analyze (default: all)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if len(args.files) != len(args.labels):
        print("ERROR: Number of files must match number of labels")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    print("="*100)
    print("SUBGRAPH STRUCTURE VISUALIZATION")
    print("="*100)
    
    # Load KG data
    kg_data = load_kg_data(args.dataset_path)
    if kg_data is None:
        return
    
    # Analyze all files
    all_results = {}
    all_data = {}
    
    for filepath, label in zip(args.files, args.labels):
        metrics_list = analyze_explanation_file(filepath, kg_data, args.sample_size)
        all_results[label] = metrics_list
        all_data[label] = torch.load(filepath, map_location='cpu')
    
    print(f"\n{'='*100}")
    print("GENERATING VISUALIZATIONS")
    print('='*100)
    
    # 1. Summary comparison
    print("\n1. Creating summary comparison...")
    create_summary_comparison(all_results, output_dir / 'summary_comparison.png')
    
    # 2. Distribution plots
    print("\n2. Creating distribution plots...")
    create_distribution_comparison(all_results, 'density', 
                                  'Graph Density Distribution',
                                  'Density',
                                  output_dir / 'dist_density.png')
    
    create_distribution_comparison(all_results, 'num_nodes',
                                  'Number of Nodes Distribution',
                                  'Number of Nodes',
                                  output_dir / 'dist_num_nodes.png')
    
    create_distribution_comparison(all_results, 'avg_clustering',
                                  'Clustering Coefficient Distribution',
                                  'Clustering Coefficient',
                                  output_dir / 'dist_clustering.png')
    
    # 3. Box plots
    print("\n3. Creating box plots...")
    create_boxplot_comparison(all_results, 'density',
                             'Density Comparison',
                             'Density',
                             output_dir / 'box_density.png')
    
    create_boxplot_comparison(all_results, 'num_components',
                             'Number of Components Comparison',
                             'Number of Components',
                             output_dir / 'box_components.png')
    
    create_boxplot_comparison(all_results, 'relation_diversity',
                             'Relation Diversity Comparison',
                             'Relation Diversity',
                             output_dir / 'box_relation_diversity.png')
    
    # 4. Connectivity comparison
    print("\n4. Creating connectivity comparison...")
    create_connectivity_comparison(all_results, output_dir / 'connectivity_comparison.png')
    
    # 5. Scatter plots
    print("\n5. Creating scatter plots...")
    create_scatter_comparison(all_results, 'num_nodes', 'density',
                             'Number of Nodes', 'Density',
                             'Nodes vs Density',
                             output_dir / 'scatter_nodes_density.png')
    
    create_scatter_comparison(all_results, 'density', 'avg_clustering',
                             'Density', 'Clustering Coefficient',
                             'Density vs Clustering',
                             output_dir / 'scatter_density_clustering.png')
    
    # 6. Sample subgraph visualizations
    print("\n6. Creating sample subgraph visualizations...")
    for label in args.labels:
        visualize_sample_subgraph(all_results[label], kg_data, all_data[label],
                                 label, output_dir / f'sample_subgraph_{label}.png')
    
    print(f"\n{'='*100}")
    print(f"VISUALIZATION COMPLETE")
    print(f"All plots saved to: {output_dir.absolute()}")
    print('='*100)


if __name__ == "__main__":
    main()
