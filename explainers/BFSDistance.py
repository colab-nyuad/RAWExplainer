import torch
from torch_scatter import scatter_min
from explainers.BaseExplainer import BaseExplainer
from explainers.data_util import batched_k_hop_subgraph


class BFSDistance(BaseExplainer):
    """
    Baseline explainer that selects edges based on BFS distance from central nodes.
    
    Unlike EgoNet/RandomWalk which use NBFNet's top-k predictions as central nodes,
    this baseline uses ONLY the query (head, ground_truth_tail) as central nodes.
    Edges closer to central nodes (smaller hop distance) are prioritized.
    
    This provides a model-agnostic baseline for comparison.
    """

    def __init__(
        self,
        model_to_explain,
        hops=None,
        topk_tails=1,                                   # Ignored: we use ground truth tail only
        eval_mask_type="hard_edge_mask_top_k",
        ego_network=True,
        factual=True,
        counter_factual=False,
        model_to_evaluate=None,
        use_ground_truth_only=True,                     # Key difference from EgoNet
        **kwargs,
    ):
        super().__init__(
            model_to_explain=model_to_explain,
            topk_tails=topk_tails,
            eval_mask_type=eval_mask_type,
            ego_network=ego_network,
            factual=factual,
            counter_factual=counter_factual,
            model_to_evaluate=model_to_evaluate,
            **kwargs,
        )

        self._config_hops = hops
        self.use_ground_truth_only = use_ground_truth_only
        
        if model_to_explain is not None:
            model_hops = model_to_explain.hops
            if hops is not None:
                if hops > model_hops:
                    print(f"WARNING: BFSDistance hops ({hops}) > model hops ({model_hops})")
                    print(f"         Clamping to model hops = {model_hops}")
                    self.hops = model_hops
                else:
                    self.hops = hops
                    print(f"BFSDistance: Using {hops} hops (model has {model_hops})")
            else:
                self.hops = model_hops
                print(f"BFSDistance: Using model default hops = {model_hops}")

        if not self.ego_network:
            print("WARNING: BFSDistance explainer should be used with ego_network=True")

        if self.factual or self.counter_factual:
            print("factual / counter_factual objectives not meaningful for BFSDistance")

        # Store edge distances for selection
        self.edge_distances = None

    def reset_edge_distances(self):
        """Reset the cached edge distances for a new instance."""
        self.edge_distances = None

    def train_mask(self, data, batch):
        """
        Compute edge distances from central nodes using BFS.
        Edges are scored inversely by their distance (closer = higher score).
        """
        # Get edge distances that were computed during ego network extraction
        # We'll use a simple heuristic: edges touching closer nodes get higher priority
        
        # For now, compute distance-based scores from the edge endpoints
        # Edges where both endpoints are at hop 1 get score 2
        # Edges where one endpoint is at hop 1 get score 1.5
        # Edges where both endpoints are at hop 2 get score 1, etc.
        
        num_edges = data.edge_type.shape[0]
        
        # If we have edge_distance info from batched_k_hop_subgraph, use it
        if hasattr(data, 'edge_distance') and data.edge_distance is not None:
            # Convert distance to score: closer edges get higher scores
            # Use negative distance so smaller distance = higher score
            max_dist = data.edge_distance.max()
            self.edge_distances = max_dist - data.edge_distance + 1
        else:
            # Fallback: compute based on node distances from central nodes
            # For each edge, score = avg distance of src and tgt from central nodes
            self.edge_distances = self._compute_edge_scores_from_structure(data)

    def _compute_edge_scores_from_structure(self, data):
        """
        Truly BFS: Computes exact hop distances from central nodes using vectorized scatter operations.
        Scores edges based on the proximity of their endpoints to the center.
        """
        device = data.edge_index.device
        num_nodes = data.subgraph_num_nodes.sum() # Total nodes across all subgraphs in batch
        num_edges = data.edge_index.size(1)
        
        # 1. Initialize node distances
        # We use a large value (self.hops + 1) to represent infinity
        inf = self.hops + 1
        node_dist = torch.full((num_nodes,), inf, device=device, dtype=torch.long)
        
        # Get central node indices adjusted for batch offsets
        # data.central_node_index is typically (batch, nodes_per_triple)
        # We flatten them to the global subgraph indexing
        batch_offsets = torch.cat([torch.tensor([0], device=device), data.subgraph_num_nodes.cumsum(0)[:-1]])
        flat_central_nodes = (data.central_node_index + batch_offsets.unsqueeze(1)).flatten()
        node_dist[flat_central_nodes] = 0

        # 2. Vectorized BFS (Bellman-Ford style propagation)
        # We only need to run this for 'self.hops' iterations
        src, tgt = data.edge_index
        for _ in range(self.hops):
            # Check distance from src to tgt
            new_dist_tgt, _ = scatter_min(node_dist[src] + 1, tgt, dim=0, dim_size=num_nodes)
            # Check distance from tgt to src (assuming undirected/bidirectional for BFS)
            new_dist_src, _ = scatter_min(node_dist[tgt] + 1, src, dim=0, dim_size=num_nodes)
            
            # Update distances with the minimum found in this hop
            node_dist = torch.minimum(node_dist, torch.minimum(new_dist_tgt, new_dist_src))

        # 3. Calculate Edge Scores
        # An edge is as "close" as its closest endpoint
        edge_dist = torch.minimum(node_dist[src], node_dist[tgt])
        
        # Convert distance to score: Higher score = Closer to center
        # Score range: (self.hops + 1) for central edges down to 1 for furthest edges
        edge_scores = (inf - edge_dist).float()

        # 4. Tie-breaking (Deterministic)
        # Use edge_index to break ties so the same edges are always picked if distances match
        tie_breaker = (src.float() + tgt.float()) * 1e-6
        edge_scores += tie_breaker

        return edge_scores

    @torch.no_grad()
    def evaluate_mask(self, data, batch):
        """
        Generate explanation by selecting edges closest to central nodes.
        Uses pre-computed distance scores from train_mask.
        """
        assert (
            "hard" in self.eval_mask_type
        ), "Only evaluation of hard masks supported for BFSDistance."

        # Use cached distance scores, or compute if not available
        if self.edge_distances is None:
            self.train_mask(data, batch)

        mask = self.edge_distances

        # Transform mask to get top-k edges
        data, mask, node_mask, edge_mask, num_edges, num_nodes = self.transform_mask(
            data, mask
        )

        # Get edge weight for evaluation
        edge_weight = self.get_edge_weight_eval(data, mask)

        # Evaluate on the masked subgraph
        masked_pred = self.eval_model(data, batch, edge_weight=edge_weight)
        assert not torch.any(torch.isnan(masked_pred))
        return masked_pred, node_mask, edge_mask, num_edges, num_nodes
