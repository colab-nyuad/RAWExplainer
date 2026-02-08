import os
import sys
import random
import torch
import csv, time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nbfnet import util

separator = ">" * 30
line = "-" * 30

if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    # Explainer Params (focused around the winning config)
    cfg.explainer.size_reg = random.randint(4, 6)                 # winner: 5
    cfg.explainer.ent_reg = random.randint(4, 6)                  # winner: 5
    cfg.explainer.sample_bias = random.choice([0.25, 0.3, 0.35, 0.4])  # winner: 0.3
    cfg.explainer.temp_start = random.choice([1, 2])              # winner: 1
    # should be lower than or equal to temp start (winner used 1)
    cfg.explainer.temp_end = 1
    cfg.explainer.num_mlp_layer = random.choice([2, 3, 4])        # winner: 3
    cfg.explainer.use_default_aggr = random.choice([False])       # fixed False
    cfg.explainer.topk_tails = random.randint(2, 3)               # winner: 3
    cfg.explainer.ego_network = random.choice([False, True])      # winner: False
    cfg.optimizer.lr = random.choice([1e-4, 3e-4, 5e-4, 8e-4])    # winner: 5e-4
    cfg.train.num_epoch = random.choice([5])                      # fixed 5
    # RW params (around winning values)
    cfg.explainer.random_walk_loss = True
    cfg.explainer.use_networkx_rw = True
    cfg.explainer.adj_aggr = random.choice(["mean", "sum"])       # winner: "mean"
    cfg.explainer.teleport_prob = random.choice([0.03, 0.05, 0.08, 0.1])  # winner: 0.05
    cfg.explainer.rw_topk_node = random.choice([50, 100, 200])    # winner: 100
    cfg.explainer.reg_loss_inside = random.choice([0.5, 1.0, 1.5])    # winner: 1
    cfg.explainer.reg_loss_outside = random.choice([0.5, 1.0, 1.5])   # winner: 1
    cfg.explainer.use_teleport_adj = random.choice([True])        # fixed True
    # Path Params (winner had with_path_loss=False, reg_path_loss≈1 but unused)
    cfg.explainer.with_path_loss = random.choice([False, True])
    cfg.explainer.reg_path_loss = random.randint(0, 2)
    cfg.explainer.max_path_length = 3
    # Optim Params
    cfg.explainer.factual = True
    cfg.explainer.expl_gnn_model = False

    cfg.wandb.name = cfg.wandb.name + f"_{args.hyper_run_id}"
    torch.save(cfg, f"hyperparam_cfg_rawexplainer_diff_k_2.pt")

    # persist sampled hyperparameters to CSV for reproducibility
    out_dir = os.path.join(cfg.output_dir, "hyper_cfgs")
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"hyperparams_log_2_sample.csv")

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