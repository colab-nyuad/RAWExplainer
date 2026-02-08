import torch
from explainers.BaseExplainer import BaseExplainer
from explainers.data_util import recreate_data_object


class EgoNet(BaseExplainer):
    """
    Baseline explainer that randomly samples edges from the ego network.
    Similar structure to RandomWalk but uses random edge selection instead of PageRank.
    
    The ego network is pre-extracted during data preparation with hops = len(hidden_dims).
    This explainer can optionally use fewer hops by filtering edges based on distance.

    NOTE: EgoNet uses NBFNet (like RandomWalk) a guide to generate top-k predictions,
          and then it uses random edge selection to generate the explanation subgraphs.
    """

    def __init__(
        self,
        model_to_explain,
        hops=None,                              # Optional: limit hops (must be <= model hops)
        topk_tails=1,
        eval_mask_type="hard_edge_mask_top_k",
        ego_network=True,
        factual=True,
        counter_factual=False,
        model_to_evaluate=None,
        use_ground_truth_only=False,            # If True: use only (head, gt_tail), skip NBFNet predictions
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
        
        # Model-agnostic mode: use only ground truth tail, no NBFNet predictions
        self.use_ground_truth_only = use_ground_truth_only
        if use_ground_truth_only:
            print("EgoNet: Model-agnostic mode enabled (using ground truth tail only)")

        # Store the configured hops limit
        self._config_hops = hops
        
        # Validate and set hops
        if model_to_explain is not None:
            model_hops = model_to_explain.hops
            if hops is not None:
                if hops > model_hops:
                    print(f"WARNING: EgoNet hops ({hops}) > model hops ({model_hops})")
                    print(f"         Clamping to model hops = {model_hops}")
                    self.hops = model_hops
                else:
                    self.hops = hops
                    print(f"EgoNet: Using {hops} hops (model has {model_hops})")
            else:
                self.hops = model_hops
                print(f"EgoNet: Using model default hops = {model_hops}")
        
        if not self.ego_network:
            print("WARNING: EgoNet explainer should be used with ego_network=True")

        if self.factual or self.counter_factual:
            print("factual / counter_factual objectives not meaningful for EgoNet")

        # Store random edge scores for consistency between train and eval
        self.edge_scores = None

    def reset_edge_scores(self):
        """Reset the cached edge scores for a new instance."""
        self.edge_scores = None

    def train_mask(self, data, batch):
        """
        'Training' for EgoNet: simply generate random scores for all edges.
        These scores are cached and used during evaluation.
        """
        # Generate random scores for all edges in the ego network
        num_edges = data.edge_type.shape[0]
        self.edge_scores = torch.rand(num_edges, device=data.edge_type.device)

    @torch.no_grad()
    def evaluate_mask(self, data, batch):
        """
        Generate explanation by randomly selecting edges from the ego network.
        Uses the pre-computed random scores from train_mask.
        
        Returns:
            masked_pred: predictions on the explanation subgraph
            node_mask: (batch_size, num_nodes) mask of nodes in explanation
            edge_mask: (batch_size, num_edges) mask of edges in explanation
            num_edges: number of edges per sample
            num_nodes: number of nodes per sample
        """
        assert (
            "hard" in self.eval_mask_type
        ), "Only evaluation of hard masks supported for EgoNet."

        # Use cached random scores, or generate new ones if not available
        if self.edge_scores is None:
            num_edges = data.edge_type.shape[0]
            self.edge_scores = torch.rand(num_edges, device=data.edge_type.device)

        mask = self.edge_scores

        # Transform mask to get top-k edges (uses BaseExplainer's transform_mask)
        data, mask, node_mask, edge_mask, num_edges, num_nodes = self.transform_mask(
            data, mask
        )

        # Get edge weight for evaluation
        edge_weight = self.get_edge_weight_eval(data, mask)

        # Evaluate on the masked subgraph
        masked_pred = self.eval_model(data, batch, edge_weight=edge_weight)
        assert not torch.any(torch.isnan(masked_pred))
        return masked_pred, node_mask, edge_mask, num_edges, num_nodes