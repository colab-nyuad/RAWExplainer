#!/usr/bin/env python3
"""
Compare Explanation Subgraph Structures

This script analyzes the structural properties of explanatory subgraphs from two
explanation files, focusing on valid (non-padding) edges.

Usage:
    python compare_subgraph_structure.py [--file1 PATH] [--file2 PATH] [--dataset-path PATH]
"""

import torch
import argparse
import numpy as np
from collections import defaultdict, Counter
import networkx as nx
from pathlib import Path


def load_kg_data(dataset_path):
    """Load the knowledge graph dataset to map edge IDs to actual triples"""
    print(f"\nLoading knowledge graph from: {dataset_path}")
    
    # Load the processed dataset
    processed_file = Path(dataset_path) / 'processed' / 'data.pt'
    
    if not processed_file.exists():
        print(f"ERROR: Dataset file not found: {processed_file}")
        return None
    
    data_list = torch.load(processed_file, map_location='cpu')
    # The file contains [train_data, valid_data, test_data]
    train_data = data_list[0]
    
    # Calculate num_relations from edge_type
    num_relations = train_data.edge_type.max().item() + 1
    
    print(f"Knowledge Graph loaded:")
    print(f"  Num nodes: {train_data.num_nodes}")
    print(f"  Num edges: {train_data.edge_index.shape[1]}")
    print(f"  Num relations: {num_relations}")
    
    return train_data


def get_valid_edges(explanation_row):
    """Extract valid (non -1) edge IDs from an explanation row"""
    query = explanation_row[:2]  # [head, relation]
    edge_ids = explanation_row[2:]  # Rest are edge IDs
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
        return {
            'num_nodes': 0,
            'num_edges': 0,
            'is_empty': True
        }
    
    metrics['num_nodes'] = G.number_of_nodes()
    metrics['num_edges'] = G.number_of_edges()
    metrics['is_empty'] = False
    
    # Density
    if G.number_of_nodes() > 1:
        metrics['density'] = nx.density(G)
    else:
        metrics['density'] = 0
    
    # Connected components (treating as undirected for this)
    G_undirected = G.to_undirected()
    metrics['num_components'] = nx.number_connected_components(G_undirected)
    metrics['is_connected'] = nx.is_connected(G_undirected)
    
    # Largest component size
    if metrics['num_components'] > 0:
        largest_cc = max(nx.connected_components(G_undirected), key=len)
        metrics['largest_component_size'] = len(largest_cc)
        metrics['largest_component_ratio'] = len(largest_cc) / G.number_of_nodes()
    else:
        metrics['largest_component_size'] = 0
        metrics['largest_component_ratio'] = 0
    
    # Degree distribution
    in_degrees = [d for n, d in G.in_degree()]
    out_degrees = [d for n, d in G.out_degree()]
    
    if in_degrees:
        metrics['avg_in_degree'] = np.mean(in_degrees)
        metrics['max_in_degree'] = np.max(in_degrees)
        metrics['avg_out_degree'] = np.mean(out_degrees)
        metrics['max_out_degree'] = np.max(out_degrees)
    
    # Diameter (for largest component if not connected)
    try:
        if metrics['is_connected']:
            metrics['diameter'] = nx.diameter(G_undirected)
        else:
            # Compute for largest component
            largest_cc_subgraph = G_undirected.subgraph(largest_cc)
            if largest_cc_subgraph.number_of_nodes() > 0:
                metrics['diameter'] = nx.diameter(largest_cc_subgraph)
            else:
                metrics['diameter'] = 0
    except:
        metrics['diameter'] = -1  # Could not compute
    
    # Average clustering coefficient
    try:
        metrics['avg_clustering'] = nx.average_clustering(G_undirected)
    except:
        metrics['avg_clustering'] = 0
    
    # Relation diversity
    relations = [data['relation'] for _, _, data in G.edges(data=True)]
    metrics['num_unique_relations'] = len(set(relations))
    metrics['relation_diversity'] = len(set(relations)) / len(relations) if relations else 0
    
    return metrics


def analyze_explanation_file(filepath, kg_data, sample_size=None):
    """Analyze all explanations in a file"""
    print(f"\n{'='*100}")
    print(f"Analyzing: {filepath}")
    print('='*100)
    
    data = torch.load(filepath, map_location='cpu')
    
    num_samples = data.shape[0]
    if sample_size:
        num_samples = min(num_samples, sample_size)
    
    print(f"Total samples: {data.shape[0]}, analyzing: {num_samples}")
    
    results = {
        'num_edges_per_sample': [],
        'num_nodes_per_sample': [],
        'density_per_sample': [],
        'diameter_per_sample': [],
        'num_components_per_sample': [],
        'is_connected_per_sample': [],
        'largest_component_ratio': [],
        'avg_clustering': [],
        'num_unique_relations': [],
        'relation_diversity': [],
        'empty_explanations': 0,
        'all_metrics': []
    }
    
    for i in range(num_samples):
        query, valid_edge_ids = get_valid_edges(data[i])
        
        if len(valid_edge_ids) == 0:
            results['empty_explanations'] += 1
            continue
        
        # Build subgraph
        G = build_subgraph_from_edges(valid_edge_ids, kg_data)
        metrics = compute_graph_metrics(G)
        
        # Store metrics
        if not metrics.get('is_empty', False):
            results['num_edges_per_sample'].append(metrics['num_edges'])
            results['num_nodes_per_sample'].append(metrics['num_nodes'])
            results['density_per_sample'].append(metrics['density'])
            results['diameter_per_sample'].append(metrics['diameter'])
            results['num_components_per_sample'].append(metrics['num_components'])
            results['is_connected_per_sample'].append(1 if metrics['is_connected'] else 0)
            results['largest_component_ratio'].append(metrics['largest_component_ratio'])
            results['avg_clustering'].append(metrics['avg_clustering'])
            results['num_unique_relations'].append(metrics['num_unique_relations'])
            results['relation_diversity'].append(metrics['relation_diversity'])
            results['all_metrics'].append(metrics)
    
    return results


def print_statistics(results, file_name):
    """Print statistics for a file's results"""
    print(f"\n{'='*100}")
    print(f"STATISTICS: {file_name}")
    print('='*100)
    
    total = len(results['num_edges_per_sample']) + results['empty_explanations']
    
    print(f"\nOverall:")
    print(f"  Total explanations: {total}")
    print(f"  Empty explanations: {results['empty_explanations']} ({results['empty_explanations']/total*100:.2f}%)")
    print(f"  Non-empty explanations: {len(results['num_edges_per_sample'])}")
    
    if len(results['num_edges_per_sample']) == 0:
        print("\nNo valid subgraphs to analyze!")
        return
    
    print(f"\nSubgraph Size:")
    print(f"  Edges - Mean: {np.mean(results['num_edges_per_sample']):.2f}, "
          f"Std: {np.std(results['num_edges_per_sample']):.2f}, "
          f"Min: {np.min(results['num_edges_per_sample'])}, "
          f"Max: {np.max(results['num_edges_per_sample'])}")
    print(f"  Nodes - Mean: {np.mean(results['num_nodes_per_sample']):.2f}, "
          f"Std: {np.std(results['num_nodes_per_sample']):.2f}, "
          f"Min: {np.min(results['num_nodes_per_sample'])}, "
          f"Max: {np.max(results['num_nodes_per_sample'])}")
    
    print(f"\nConnectivity:")
    print(f"  Density - Mean: {np.mean(results['density_per_sample']):.4f}, "
          f"Std: {np.std(results['density_per_sample']):.4f}")
    print(f"  Connected graphs: {sum(results['is_connected_per_sample'])} "
          f"({sum(results['is_connected_per_sample'])/len(results['is_connected_per_sample'])*100:.2f}%)")
    print(f"  Num components - Mean: {np.mean(results['num_components_per_sample']):.2f}, "
          f"Max: {np.max(results['num_components_per_sample'])}")
    print(f"  Largest component ratio - Mean: {np.mean(results['largest_component_ratio']):.4f}")
    
    valid_diameters = [d for d in results['diameter_per_sample'] if d > 0]
    if valid_diameters:
        print(f"  Diameter - Mean: {np.mean(valid_diameters):.2f}, "
              f"Max: {np.max(valid_diameters)}")
    
    print(f"\nGraph Structure:")
    print(f"  Avg clustering - Mean: {np.mean(results['avg_clustering']):.4f}")
    print(f"  Unique relations - Mean: {np.mean(results['num_unique_relations']):.2f}, "
          f"Max: {np.max(results['num_unique_relations'])}")
    print(f"  Relation diversity - Mean: {np.mean(results['relation_diversity']):.4f}")


def compare_structures(results1, results2, file1_name, file2_name):
    """Compare structural differences between two sets of results"""
    print(f"\n{'='*100}")
    print("STRUCTURAL COMPARISON")
    print('='*100)
    
    def compare_metric(name, list1, list2):
        if len(list1) == 0 or len(list2) == 0:
            print(f"\n{name}: Cannot compare (one or both lists empty)")
            return
        
        mean1 = np.mean(list1)
        mean2 = np.mean(list2)
        diff = mean2 - mean1
        diff_pct = (diff / mean1 * 100) if mean1 != 0 else 0
        
        print(f"\n{name}:")
        print(f"  {file1_name}: {mean1:.4f}")
        print(f"  {file2_name}: {mean2:.4f}")
        print(f"  Difference: {diff:+.4f} ({diff_pct:+.2f}%)")
    
    compare_metric("Average Edges per Subgraph", 
                   results1['num_edges_per_sample'], 
                   results2['num_edges_per_sample'])
    
    compare_metric("Average Nodes per Subgraph", 
                   results1['num_nodes_per_sample'], 
                   results2['num_nodes_per_sample'])
    
    compare_metric("Graph Density", 
                   results1['density_per_sample'], 
                   results2['density_per_sample'])
    
    compare_metric("Number of Components", 
                   results1['num_components_per_sample'], 
                   results2['num_components_per_sample'])
    
    compare_metric("Largest Component Ratio", 
                   results1['largest_component_ratio'], 
                   results2['largest_component_ratio'])
    
    compare_metric("Average Clustering Coefficient", 
                   results1['avg_clustering'], 
                   results2['avg_clustering'])
    
    compare_metric("Relation Diversity", 
                   results1['relation_diversity'], 
                   results2['relation_diversity'])
    
    # Connected graphs comparison
    conn1_pct = sum(results1['is_connected_per_sample']) / len(results1['is_connected_per_sample']) * 100
    conn2_pct = sum(results2['is_connected_per_sample']) / len(results2['is_connected_per_sample']) * 100
    
    print(f"\nConnected Graphs (%):")
    print(f"  {file1_name}: {conn1_pct:.2f}%")
    print(f"  {file2_name}: {conn2_pct:.2f}%")
    print(f"  Difference: {conn2_pct - conn1_pct:+.2f}%")
    
    # Empty explanations
    total1 = len(results1['num_edges_per_sample']) + results1['empty_explanations']
    total2 = len(results2['num_edges_per_sample']) + results2['empty_explanations']
    empty1_pct = results1['empty_explanations'] / total1 * 100 if total1 > 0 else 0
    empty2_pct = results2['empty_explanations'] / total2 * 100 if total2 > 0 else 0
    
    print(f"\nEmpty Explanations (%):")
    print(f"  {file1_name}: {empty1_pct:.2f}%")
    print(f"  {file2_name}: {empty2_pct:.2f}%")
    print(f"  Difference: {empty2_pct - empty1_pct:+.2f}%")


def main():

    # Default paths
    DEFAULT_FILE1 = "/scratch/oge208/Graph-Transformer/NBFNet-PyG/explanation/RAWExplainer/NBFNet/FB15k-237/2025-03-14-23-02-33-hyp-ola1_4/test_explanations_factual_eval_hard_edge_mask_top_k_100.pt"
    # DEFAULT_FILE2 = "/scratch/oge208/Graph-Transformer/NBFNet-PyG/explanation/RAWExplainer/NBFNet/FB15k-237/2026-01-31-20-10-35-hyp1-0.5lr-adamw-mediocre/test_explanations_factual_eval_hard_edge_mask_top_k_25.pt"
    # DEFAULT_FILE2 = "/scratch/oge208/Graph-Transformer/NBFNet-PyG/explanation/PGExplainer/NBFNet/FB15k-237/finetune/test_explanations_factual_eval_hard_edge_mask_top_k_100.pt"
    DEFAULT_FILE2 = "/scratch/sl9030/gml/Graph-Transformer/NBFNet-PyG/explanation/RAWExplainer/NBFNet/FB15k-237/2026-02-02-17-00-13-var44-potential/test_explanations_factual_eval_hard_edge_mask_top_k_100.pt"
    DEFAULT_DATASET = "/scratch/oge208/Graph-Transformer/NBFNet-PyG/datasets/knowledge_graphs/FB15k-237"
    
    parser = argparse.ArgumentParser(
        description='Compare subgraph structures in explanation files',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--file1', default=DEFAULT_FILE1,
                        help=f'First explanation file (default: {DEFAULT_FILE1})')
    parser.add_argument('--file2', default=DEFAULT_FILE2,
                        help=f'Second explanation file (default: {DEFAULT_FILE2})')
    parser.add_argument('--dataset-path', default=DEFAULT_DATASET,
                        help=f'Path to dataset directory (default: {DEFAULT_DATASET})')
    parser.add_argument('--sample-size', type=int, default=None,
                        help='Number of samples to analyze (default: all)')
    
    args = parser.parse_args()
    
    print("="*100)
    print("EXPLANATION SUBGRAPH STRUCTURE COMPARISON")
    print("="*100)
    
    # Load KG data
    kg_data = load_kg_data(args.dataset_path)
    if kg_data is None:
        return
    
    # Analyze both files
    results1 = analyze_explanation_file(args.file1, kg_data, args.sample_size)
    results2 = analyze_explanation_file(args.file2, kg_data, args.sample_size)
    
    # Print statistics
    print_statistics(results1, "File 1")
    print_statistics(results2, "File 2")
    
    # Compare
    compare_structures(results1, results2, "File 1", "File 2")
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()
