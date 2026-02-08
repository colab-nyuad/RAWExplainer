import torch
from torch_scatter import scatter_min, scatter
from explainers.BaseExplainer import BaseExplainer
from explainers.data_util import recreate_data_object, counter_factual_edge_filter


class BFSNode(BaseExplainer):
    """
    Model-agnostic BFS-based explainer that selects NODES by BFS distance from central nodes,
    then takes the induced edges within those nodes.
    
    This mirrors the RandomWalk approach (which ranks nodes by PageRank, then selects induced edges),
    but uses BFS distance instead of PageRank for node ranking.
    
    Algorithm:
    1. Compute BFS distance from central nodes (head, ground_truth_tail) for all nodes
    2. Rank nodes by distance (closer nodes get higher rank)
    3. Use binary search to find how many top-ranked nodes to include such that 
       the induced edges fit the budget
    4. Select edges where BOTH endpoints are in the selected node set
    """

    def __init__(
        self,
        model_to_explain,
        hops=None,
        topk_tails=1,                                   
        eval_mask_type="hard_edge_mask_top_k",
        ego_network=False,                              # Use full graph
        factual=True,
        counter_factual=False,
        model_to_evaluate=None,
        use_ground_truth_only=False,                    # If True: Model-agnostic
        use_head_only=False,                            # If True: only head, no GT bias
        threshold=0,
        keep_edge_weight=False,
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
            keep_edge_weight=keep_edge_weight,
            **kwargs,
        )

        self._config_hops = hops
        self.use_ground_truth_only = use_ground_truth_only
        self.use_head_only = use_head_only
        self.threshold = threshold
        
        if model_to_explain is not None:
            model_hops = model_to_explain.hops
            if hops is not None:
                self.hops = min(hops, model_hops)
            else:
                self.hops = model_hops
            print(f"BFSNode: Using {self.hops} hops for BFS exploration")
        else:
            self.hops = hops or 6

        if use_head_only:
            print("BFSNode: Head-only mode - using ONLY head as central node (no GT bias)")
        elif use_ground_truth_only:
            print("BFSNode: Model-agnostic mode - using only (head, ground_truth_tail) as central nodes")

        if self.factual or self.counter_factual:
            print("factual / counter_factual objectives not meaningful for BFSNode")

        # Node distances (computed in train_mask, used in evaluate_mask)
        self.node_dist = None

    def reset_node_dist(self):
        """Reset the cached node distances for a new instance."""
        self.node_dist = None

    def train_mask(self, data, batch):
        """
        Compute node distances from central nodes using BFS.
        Nodes closer to central nodes get higher priority (lower distance = higher rank).
        """
        self.node_dist = self._compute_bfs_node_distances(data)

    def _compute_bfs_node_distances(self, data):
        """
        Compute BFS-based scores for node ranking.
        
        Head-only mode: Simple BFS from head - nodes closer to head get higher scores.
        
        Head+Tail mode: Score by SUM of distances to BOTH, prioritizing nodes on 
        connecting paths (lower sum = higher score).
        """
        device = data.edge_index.device
        num_nodes = data.subgraph_num_nodes.sum()
        batch_size = data.subgraph_num_nodes.size(0)
        
        max_dist = self.hops + 1
        
        # Get central node indices adjusted for batch offsets
        batch_offsets = torch.cat([
            torch.tensor([0], device=device), 
            data.subgraph_num_nodes.cumsum(0)[:-1]
        ])
        
        src, tgt = data.edge_index
        
        if self.use_head_only:
            # Head-only mode: simple BFS from head
            # central_node_index is (batch_size, 1) with just the head
            heads = data.central_node_index[:, 0] + batch_offsets
            
            dist_from_head = torch.full((num_nodes,), max_dist, device=device, dtype=torch.float32)
            dist_from_head[heads] = 0
            
            for _ in range(self.hops):
                new_dist_tgt, _ = scatter_min(dist_from_head[src] + 1, tgt, dim=0, dim_size=num_nodes)
                new_dist_src, _ = scatter_min(dist_from_head[tgt] + 1, src, dim=0, dim_size=num_nodes)
                dist_from_head = torch.minimum(dist_from_head, torch.minimum(new_dist_tgt, new_dist_src))
            
            # Score = max_dist - distance (closer = higher score)
            node_scores = (max_dist - dist_from_head).float()
            node_scores[heads] = float('inf')  # Head always highest priority
            
        else:
            # Head+Tail mode: prioritize nodes on paths between them
            # central_node_index is (batch_size, 2) where [:, 0] = head, [:, 1] = tail
            heads = data.central_node_index[:, 0] + batch_offsets
            tails = data.central_node_index[:, 1] + batch_offsets
            
            # BFS from HEAD
            dist_from_head = torch.full((num_nodes,), max_dist, device=device, dtype=torch.float32)
            dist_from_head[heads] = 0
            
            for _ in range(self.hops):
                new_dist_tgt, _ = scatter_min(dist_from_head[src] + 1, tgt, dim=0, dim_size=num_nodes)
                new_dist_src, _ = scatter_min(dist_from_head[tgt] + 1, src, dim=0, dim_size=num_nodes)
                dist_from_head = torch.minimum(dist_from_head, torch.minimum(new_dist_tgt, new_dist_src))
            
            # BFS from TAIL
            dist_from_tail = torch.full((num_nodes,), max_dist, device=device, dtype=torch.float32)
            dist_from_tail[tails] = 0
            
            for _ in range(self.hops):
                new_dist_tgt, _ = scatter_min(dist_from_tail[src] + 1, tgt, dim=0, dim_size=num_nodes)
                new_dist_src, _ = scatter_min(dist_from_tail[tgt] + 1, src, dim=0, dim_size=num_nodes)
                dist_from_tail = torch.minimum(dist_from_tail, torch.minimum(new_dist_tgt, new_dist_src))
            
            # Score = negative of sum (lower sum = higher score = on path)
            path_distance = dist_from_head + dist_from_tail
            node_scores = (2 * max_dist - path_distance).float()
            
            node_scores[heads] = float('inf')
            node_scores[tails] = float('inf')
        
        return node_scores

    def _get_topk_nodes_by_score(self, data, node_dist, top_num_nodes=None, return_dense=False):
        """
        Get top-k nodes based on BFS distance scores (higher = closer to center).
        Similar to RWBaseExplainer.get_rw_topk_nodes but for BFS scores.
        """
        # Create dense node weight tensor (batch_size, max_num_nodes)
        node_weight = torch.zeros(
            (data.subgraph_num_nodes.size(0), data.max_num_nodes),
            device=node_dist.device,
            dtype=node_dist.dtype,
        ).fill_(float("-inf"))
        
        indices = torch.arange(
            node_weight.size(1), device=node_weight.device
        ).expand_as(node_weight)
        valid_nodes = indices < data.subgraph_num_nodes.unsqueeze(1)
        node_weight[valid_nodes] = node_dist.squeeze()

        if top_num_nodes is None:
            top_num_nodes = data.subgraph_num_nodes.clone()

        # Sort nodes by score (descending - higher score = closer to center)
        argsort = torch.argsort(node_weight, dim=1, descending=True)
        top_nodes = indices < top_num_nodes.unsqueeze(1)

        if return_dense:
            argsort[~top_nodes] = -1
            return argsort

        s_node_id = argsort[top_nodes]
        node_batch = torch.arange(
            node_weight.size(0), device=node_weight.device
        ).repeat_interleave(top_num_nodes)

        return s_node_id, node_batch

    def _efficient_hard_edge_masks(self, data, mask, node_dist, top_num_edges):
        """
        Use binary search to find optimal number of nodes to include such that 
        induced edges fit within budget.
        Mirrors RWBaseExplainer.efficient_hard_edge_masks_rw.
        """
        node_id = self._get_topk_nodes_by_score(
            data, node_dist, data.subgraph_num_nodes, return_dense=True
        )
        indices = torch.arange(
            node_id.size(1), device=data.subgraph_num_nodes.device
        ).expand_as(node_id)
        
        # Each edge gets a unique ID across batches
        offset = data.edge_batch * data.num_nodes
        unique_edges = data.edge_index + offset

        def get_edge_mask(rank_count):
            # Get selected nodes for given rank count
            selected_node_mask = indices < rank_count.unsqueeze(1)
            selected_node_id = node_id[selected_node_mask]
            node_batch = torch.arange(
                node_id.size(0), device=node_id.device
            ).repeat_interleave(rank_count)
            selected_nodes = selected_node_id + node_batch * data.num_nodes

            # Get edges where BOTH endpoints are in selected nodes
            edge_mask = torch.all(torch.isin(unique_edges, selected_nodes), dim=0)
            # Apply threshold
            threshold_mask = mask >= self.threshold
            edge_mask = torch.logical_and(edge_mask, threshold_mask)
            return edge_mask

        def get_num_edges(rank_count):
            edge_mask = get_edge_mask(rank_count)
            num_edges = scatter(
                edge_mask.to(torch.long),
                data.edge_batch,
                dim=0,
                dim_size=data.subgraph_num_edges.size(0),
                reduce="sum",
            )
            return num_edges

        # Binary search for optimal rank
        rank_low = torch.zeros_like(data.subgraph_num_nodes)
        rank_high = data.subgraph_num_nodes.clone()
        best = torch.zeros_like(data.subgraph_num_nodes)

        while torch.any(rank_low <= rank_high):
            rank_mid = (rank_high + rank_low) // 2
            num_edges = get_num_edges(rank_mid)

            met_quota = num_edges > top_num_edges
            rank_high[met_quota] = rank_mid[met_quota] - 1
            best[~met_quota] = rank_mid[~met_quota]
            rank_low[~met_quota] = rank_mid[~met_quota] + 1

        edge_mask = get_edge_mask(best)
        mask = mask[edge_mask]
        edges = data.edge_index[:, edge_mask]
        edge_type = data.edge_type[edge_mask]
        edge_batch = data.edge_batch[edge_mask]

        # Rebuild data object with selected edges
        if self.factual_eval:
            data = recreate_data_object(data, edges, edge_type, edge_batch)
        elif self.counter_factual_eval:
            data = counter_factual_edge_filter(data, edge_mask=edge_mask)

        node_mask, edge_mask = self.get_node_mask_and_edge_mask(
            data, edge_batch, edges, edge_type
        )
        
        if not self.keep_edge_weight:
            mask = mask.fill_(1)
            
        num_edges = edge_mask.sum(dim=-1)
        num_nodes = node_mask.sum(dim=-1)
        
        return data, mask, node_mask, edge_mask, num_edges, num_nodes

    @torch.no_grad()
    def evaluate_mask(self, data, batch):
        """
        Generate explanation by:
        1. Using BFS node distances (computed in train_mask)
        2. Selecting nodes closest to center
        3. Taking induced edges within those nodes
        """
        assert (
            "hard" in self.eval_mask_type
        ), "Only evaluation of hard masks supported for BFSNode."

        # Use cached node distances, or compute if not available
        if self.node_dist is None:
            self.train_mask(data, batch)

        mask = torch.ones_like(data.edge_type).to(torch.float32)

        # Get hard edge mask budget
        top_num_edges = torch.tensor(
            [self.hard_edge_mask_top_k] * data.subgraph_num_edges.size(0),
            device=data.edge_index.device
        )

        # Select edges using node-based selection
        data, mask, node_mask, edge_mask, num_edges, num_nodes = (
            self._efficient_hard_edge_masks(data, mask, self.node_dist, top_num_edges)
        )

        # Get edge weight for evaluation
        edge_weight = self.get_edge_weight_eval(data, mask)

        # Evaluate on the masked subgraph
        masked_pred = self.eval_model(data, batch, edge_weight=edge_weight)
        assert not torch.any(torch.isnan(masked_pred))
        return masked_pred, node_mask, edge_mask, num_edges, num_nodes
