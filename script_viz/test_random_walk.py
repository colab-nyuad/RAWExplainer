'''
    Test random walk on a single query from test split and visualize scores

    USAGE:
        python script_viz/test_random_walk.py --help
        python script_viz/test_random_walk.py
        python script_viz/test_random_walk.py -n 5 -r True
        python script_viz/test_random_walk.py -i 0 10 20
        python script_viz/test_random_walk.py --config config/new_explain/wn18rr-rawexplainer-dev.yaml -i 0 1 2 -n 3 -s 42
'''

import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nbfnet import tasks, util
from explainers import explainer_util, data_util
from explainers.RAWExplainer import RAWExplainer

def visualize_random_walk_scores(cfg, query_idx=0, dataset=None, model=None, eval_model=None, explainer=None):
    """
    Test random walk on a single query from test split and visualize scores
    
    Args:
        cfg: Configuration object
        query_idx: Index of the query to test (default: 0 for first query)
        dataset: Pre-loaded dataset (optional)
        model: Pre-loaded model (optional)  
        eval_model: Pre-loaded eval model (optional)
        explainer: Pre-loaded explainer (optional)
    """
    
    # * Dataset Setup *
    if dataset is None:
        print("Loading dataset...")
        dataset = util.build_dataset(cfg)
        cfg.model.num_relation = dataset.num_relations
    
    # * Model Setup *
    if model is None or eval_model is None:
        print("Loading models...")
        model, eval_model = util.build_model_expl(cfg)
    
    # * Device Setup *
    device = util.get_device(cfg)
    model = model.to(device)
    
    # * Data Setup *
    train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
    train_data = train_data.to(device)
    test_data = test_data.to(device)
    
    # * Build RAWExplainer *
    if explainer is None:
        print("Loading explainer...")
        explainer = explainer_util.build_explainer(cfg, model, eval_model)
        explainer = explainer.to(device)
    explainer.eval()
    
    print(f"Dataset: {cfg.dataset['class']}")
    print(f"Number of nodes: {train_data.num_nodes}")
    print(f"Number of edges: {train_data.edge_index.size(1)}")
    print(f"Number of relations: {dataset.num_relations}")
    
    # * Get a single test query *
    test_triplets = torch.cat([
        test_data.target_edge_index, 
        test_data.target_edge_type.unsqueeze(0)
    ]).t()
    
    if query_idx >= len(test_triplets):
        query_idx = 0
        print(f"Query index too large, using query_idx=0")
    
    query = test_triplets[query_idx:query_idx+1]  # Single query
    print(f"\nTesting query {query_idx}: {query.cpu().numpy()}")
    
    h_idx, t_idx, r_idx = query[0]
    print(f"Head: {h_idx.item()}, Tail: {t_idx.item()}, Relation: {r_idx.item()}")
    
    # * Prepare data for the query *
    batch_size = 1
    mode = torch.ones((batch_size,), dtype=torch.bool, device=device)  # Tail prediction
    
    # Convert to tail prediction format
    batch = tasks.all_negative(test_data, query)[0]  # Get tail batch
    
    # Get topk predicted tails and embeddings for the explainer
    topk_tails, node_embeds, R_embeds = explainer_util.full_graph_prediction(
        test_data, query, mode, explainer
    )
    
    heads = torch.where(mode, query[:, 0], query[:, 1])
    central_nodes = torch.cat((heads.unsqueeze(1), topk_tails), dim=-1)
    
    batch = tasks.conversion_to_tail_prediction(
        batch, explainer.eval_model.num_relation, mode
    )
    
    # Prepare data structure
    if explainer.ego_network:
        data = data_util.prepare_ego_data(test_data, central_nodes, explainer.hops)
        batch = data_util.check_if_tail_in_network(data, batch)
        batch, valid_tails = data_util.relabel_batch(data, batch)
        data = data_util.create_batched_data(data)
        node_embeds = data_util.create_s_node_embeds(data, node_embeds)
    else:
        data = data_util.prepare_full_data(test_data, len(batch), central_nodes)
        data = data_util.create_batched_data(data)
    
    # Apply masking
    data = explainer.eval_model.masking(data, batch)
    
    print(f"\nSubgraph info:")
    print(f"Number of nodes in subgraph: {data.num_nodes}")
    print(f"Number of edges in subgraph: {data.edge_index.size(1)}")
    
    # * Generate explanation mask *
    with torch.no_grad():
        explainer.eval()
        
        # Get explanation mask
        mask = explainer.get_mask(data, batch, node_embeds, R_embeds)
        
        print(f"\nMask statistics:")
        print(f"Mask shape: {mask.shape}")
        print(f"Mask min: {mask.min().item():.4f}")
        print(f"Mask max: {mask.max().item():.4f}")
        print(f"Mask mean: {mask.mean().item():.4f}")
        
        # * Perform Random Walk *
        print("\nPerforming random walk...")
        
        # Reset node distribution to avoid conflicts between queries
        if hasattr(explainer, 'node_dist'):
            explainer.node_dist = None
        
        # Set random walk parameters for testing
        explainer.random_walk_loss = True
        explainer.edge_random_walk = False  # Use node-based random walk
        
        # Print random walk parameters
        print(f"Random walk parameters:")
        print(f"  max_power_iter: {explainer.max_power_iter}")
        print(f"  teleport_prob: {explainer.teleport_prob}")
        print(f"  convergence criterion: eps = 1e-3")
        print(f"  use_teleport_adj: {explainer.use_teleport_adj}")
        
        # Call the random walk function
        result = explainer.do_random_walk(data, mask, return_loss=True, store_walk=True)
        
        # This happens when store_walk=True and node_dist was None
        if result is None:
            print("Random walk stored node distribution. Calling again to get losses...")
            rw_loss, path_loss = explainer.do_random_walk(data, mask, return_loss=True, store_walk=False)
        else:
            rw_loss, path_loss = result
        
        print(f"Random walk loss: {rw_loss.item():.4f}")
        print(f"Path loss: {path_loss.item():.4f}")
        
        # Get node distribution if available
        if hasattr(explainer, 'node_dist') and explainer.node_dist is not None:
            node_dist = explainer.node_dist
            print(f"\nNode distribution statistics:")
            print(f"Node dist shape: {node_dist.shape}")
            print(f"Node dist min: {node_dist.min().item():.4f}")
            print(f"Node dist max: {node_dist.max().item():.4f}")
            print(f"Node dist mean: {node_dist.mean().item():.4f}")
            print(f"Node dist sum: {node_dist.sum().item():.4f}")
            
            # * Visualization *
            print("\nCreating visualizations...")
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot 1: Edge mask histogram
            mask_np = mask.detach().cpu().numpy()
            axes[0, 0].hist(mask_np, bins=50, alpha=0.7, color='blue')
            axes[0, 0].set_title('Edge Mask Distribution')
            axes[0, 0].set_xlabel('Mask Value')
            axes[0, 0].set_ylabel('Frequency')
            
            # Plot 2: Node distribution histogram
            node_dist_np = node_dist.detach().cpu().numpy().flatten()  # Flatten to 1D
            axes[0, 1].hist(node_dist_np, bins=50, alpha=0.7, color='red')
            axes[0, 1].set_title('Node Distribution from Random Walk')
            axes[0, 1].set_xlabel('Node Probability')
            axes[0, 1].set_ylabel('Frequency')
            
            # Plot 3: Top nodes by random walk score
            top_k = min(20, len(node_dist_np))
            top_indices = np.argsort(node_dist_np)[-top_k:]
            top_scores = node_dist_np[top_indices]
            
            axes[1, 0].bar(range(top_k), top_scores)
            axes[1, 0].set_title(f'Top {top_k} Nodes by Random Walk Score')
            axes[1, 0].set_xlabel('Node Rank')
            axes[1, 0].set_ylabel('Random Walk Score')
            
            # Plot 4: Edge mask vs edge index
            edge_indices = np.arange(len(mask_np))
            axes[1, 1].scatter(edge_indices, mask_np, alpha=0.6, s=1)
            axes[1, 1].set_title('Edge Mask Values vs Edge Index')
            axes[1, 1].set_xlabel('Edge Index')
            axes[1, 1].set_ylabel('Mask Value')
            
            plt.tight_layout()
            
            # Create dataset-specific subfolder and save the plot
            dataset_name = cfg.dataset['class'].lower()
            plot_dir = f'plots/rw/{dataset_name}'
            os.makedirs(plot_dir, exist_ok=True)
            
            plot_filename = f'{plot_dir}/random_walk_analysis_query_{query_idx}.png'
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Visualization saved as: {plot_filename}")
            
            # Also create a detailed analysis
            print(f"\nDetailed Analysis:")
            print(f"Query: Head={h_idx.item()}, Tail={t_idx.item()}, Relation={r_idx.item()}")
            print(f"Head node RW score: {node_dist[h_idx].item():.6f}")
            print(f"Tail node RW score: {node_dist[t_idx].item():.6f}")
            
            # Find top nodes  
            node_dist_flat = node_dist.flatten()  # Use flattened tensor for indexing
            top_10_indices = torch.argsort(node_dist_flat, descending=True)[:10]
            print(f"\nTop 10 nodes by random walk score:")
            for i, node_idx in enumerate(top_10_indices):
                score = node_dist_flat[node_idx].item()
                node_type = "HEAD" if node_idx.item() == h_idx.item() else "TAIL" if node_idx.item() == t_idx.item() else "OTHER"
                print(f"  {i+1}. Node {node_idx.item()}: {score:.6f} ({node_type})")
            
            # Find top edges
            top_edge_indices = np.argsort(mask_np)[-10:][::-1]
            print(f"\nTop 10 edges by mask score:")
            edge_index_np = data.edge_index.cpu().numpy()
            for i, edge_idx in enumerate(top_edge_indices):
                score = mask_np[edge_idx]
                src, tgt = edge_index_np[0, edge_idx], edge_index_np[1, edge_idx]
                print(f"  {i+1}. Edge {edge_idx} ({src}->{tgt}): {score:.6f}")
            
        else:
            print("Node distribution not available. Random walk might not have stored results.")
    
    return explainer, data, mask, query

if __name__ == "__main__":
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test Random Walk on Knowledge Graph Queries")
    parser.add_argument("--config", "-c", default="config/new_explain/wn18rr-rawexplainer-dev.yaml",
                       help="Path to config file")
    parser.add_argument("--random_queries", "-r", type=str, choices=['True', 'False'], default='False',
                       help="Randomly select queries instead of using fixed indices (True/False, default: False)")
    parser.add_argument("--query_indices", "-i", nargs='+', type=int, default=[0, 1, 2],
                       help="List of query indices to test (default: 0 1 2)")
    parser.add_argument("--num_queries", "-n", type=int, default=3,
                       help="Number of queries to test (default: 3)")
    parser.add_argument("--seed", "-s", type=int, default=1024,
                       help="Random seed for reproducibility")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0],
                       help="GPU devices to use")
    
    args = parser.parse_args()
    
    # Convert string to boolean for random_queries
    use_random_queries = args.random_queries.lower() == 'true'
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set up variables for config loading
    vars_dict = {"gpus": args.gpus}
    cfg = util.load_config(args.config, context=vars_dict)
    
    print("Testing Random Walk on Single Query")
    print("=" * 50)
    print(f"Config: {args.config}")
    print(f"Random seed: {args.seed}")
    print(f"Number of queries to test: {args.num_queries}")
    
    # Load dataset, models, and explainer once
    print("Loading dataset and models (one time)...")
    dataset = util.build_dataset(cfg)
    cfg.model.num_relation = dataset.num_relations
    model, eval_model = util.build_model_expl(cfg)
    device = util.get_device(cfg)
    model = model.to(device)
    explainer = explainer_util.build_explainer(cfg, model, eval_model)
    explainer = explainer.to(device)
    
    # Determine query indices to test
    if use_random_queries:
        print("Using RANDOM query selection")
        test_data = dataset[2]
        test_triplets = torch.cat([
            test_data.target_edge_index, 
            test_data.target_edge_type.unsqueeze(0)
        ]).t()
        
        # Randomly select specified number of query indices
        total_queries = len(test_triplets)
        num_to_select = min(args.num_queries, total_queries)
        query_indices = torch.randperm(total_queries)[:num_to_select].tolist()
        print(f"Selected random queries: {query_indices} (out of {total_queries} total)")
        
        if len(args.query_indices) != [0, 1, 2]:
            print(f"Note: Ignoring provided query_indices {args.query_indices} due to --random_queries=True")
    else:
        # Use provided query indices (take only num_queries amount)
        query_indices = args.query_indices[:args.num_queries]
        print(f"Using FIXED query indices: {query_indices}")
    
    # Test with selected queries
    for i, query_idx in enumerate(query_indices):
        print(f"\n{'='*20} Query {i} (Index {query_idx}) {'='*20}")
        explainer_obj, data, mask, query = visualize_random_walk_scores(
                                                cfg, query_idx=query_idx, dataset=dataset, model=model, 
                                                eval_model=eval_model, explainer=explainer      )
        print(f"Successfully tested query {i} (index {query_idx})")