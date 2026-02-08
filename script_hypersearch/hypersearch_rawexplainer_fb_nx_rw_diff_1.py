import os
import sys
import random
import torch
import csv, time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nbfnet import util

separator = ">" * 30
line = "-" * 30

def pick_around(anchor, choices):
    '''
    IDEA: Add weight to the choices of parameters
          Weight is added based on RAWExplainer-Hyp-Ola1_4 configurations
    '''
    pool = []
    for v, weight in choices:
        pool += [v] * weight
    if anchor in [v for v, _ in choices]:
        pool += [anchor] * 3
    return random.choice(pool)

if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    # Explainer Params
    cfg.explainer.size_reg = pick_around(5, [(4, 1), (5, 3), (6, 1)])
    cfg.explainer.ent_reg = pick_around(4, [(3, 1), (4, 3), (5, 1)])
    cfg.explainer.sample_bias = pick_around(0.5, [(0.4, 1), (0.5, 4), (0.6, 1)])
    cfg.explainer.temp_start = pick_around(3, [(3, 3), (4, 1)])
    # should be lower than temp start
    cfg.explainer.temp_end = min(pick_around(2, [(1, 1), (2, 3)]), cfg.explainer.temp_start - 1)
    cfg.explainer.num_mlp_layer = pick_around(3, [(3, 3), (4, 1)])
    cfg.explainer.use_default_aggr = random.choice([False])
    cfg.explainer.topk_tails = pick_around(4, [(3, 1), (4, 3), (5, 1)])
    cfg.explainer.ego_network = pick_around(False, [(False, 4), (True, 1)])
    cfg.optimizer.lr = pick_around(1e-5, [(5e-6, 1), (1e-5, 4), (2e-5, 1)])
    cfg.train.num_epoch = random.choice([5])
    # RW params
    cfg.explainer.random_walk_loss = True
    cfg.explainer.use_networkx_rw = True
    cfg.explainer.adj_aggr = pick_around("max", [("max", 4), ("mean", 1), ("sum", 1)])
    cfg.explainer.teleport_prob = pick_around(0.2, [(0.15, 1), (0.2, 4), (0.25, 1)])
    cfg.explainer.rw_topk_node = pick_around(700, [(500, 1), (700, 4), (1000, 1)])
    cfg.explainer.reg_loss_inside = pick_around(1.5, [(1.0, 1), (1.5, 4), (2.0, 1)])
    cfg.explainer.reg_loss_outside = pick_around(0.75, [(0.5, 1), (0.75, 4), (1.0, 1)])
    cfg.explainer.use_teleport_adj = random.choice([True])
    # Path Params
    cfg.explainer.with_path_loss = pick_around(False, [(False, 4), (True, 1)])
    cfg.explainer.reg_path_loss = pick_around(5, [(3, 1), (4, 1), (5, 4)])
    cfg.explainer.max_path_length = 3
    # Optim Params
    cfg.explainer.factual = True
    cfg.explainer.expl_gnn_model = False

    cfg.wandb.name = cfg.wandb.name + f"_{args.hyper_run_id}"
    torch.save(cfg, f"hyperparam_cfg_rawexplainer_diff_k_1.pt")

    # persist sampled hyperparameters to CSV for reproducibility
    out_dir = os.path.join(cfg.output_dir, "hyper_cfgs")
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"hyperparams_log_1_sample.csv")

    row = {
        "run_id": args.hyper_run_id,
        # --- explainer ---
        "size_reg": cfg.explainer.size_reg,
        "ent_reg": cfg.explainer.ent_reg,
        "sample_bias": cfg.explainer.sample_bias,
        "temp_start": cfg.explainer.temp_start,
        "temp_end": cfg.explainer.temp_end,
        "num_mlp_layer": cfg.explainer.num_mlp_layer,
        "use_default_aggr": cfg.explainer.use_default_aggr,
        "topk_tails": cfg.explainer.topk_tails,
        "ego_network": cfg.explainer.ego_network,
        "adj_aggr": cfg.explainer.adj_aggr,
        "teleport_prob": cfg.explainer.teleport_prob,
        "rw_topk_node": cfg.explainer.rw_topk_node,
        "reg_loss_inside": cfg.explainer.reg_loss_inside,
        "reg_loss_outside": cfg.explainer.reg_loss_outside,
        "use_teleport_adj": cfg.explainer.use_teleport_adj,
        "with_path_loss": cfg.explainer.with_path_loss,
        "reg_path_loss": cfg.explainer.reg_path_loss,
        "max_path_length": cfg.explainer.max_path_length,
        # --- optimizer/train ---
        "lr": cfg.optimizer.lr,
        "num_epoch": cfg.train.num_epoch,
        # (optional) add static metadata if useful
        "wandb_name": cfg.wandb.name,
    }

    file_exists = os.path.isfile(out_path)
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Logged hyperparameters for run {args.hyper_run_id} -> {out_path}")
