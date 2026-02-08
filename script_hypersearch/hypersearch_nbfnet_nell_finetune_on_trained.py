import os
import sys
import random
import torch
import csv
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nbfnet import util

separator = ">" * 30
line = "-" * 30

if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    
    # Fixed
    cfg.model.message_func = "distmult"
    
    # Log the hyperparameters being used
    print(f"\n{separator}")
    print(f"HYPERPARAMETER COMBINATION (Fine-tuning):")
    print(f"hidden_dims: {cfg.model.hidden_dims} (kept from config)")
    print(f"message_func: {cfg.model.message_func} (fixed)")
    print(f"short_cut: {cfg.model.short_cut}")
    print(f"dependent: {cfg.model.dependent}")
    print(f"concat_hidden: {cfg.model.concat_hidden}")
    print(f"num_mlp_layer: {cfg.model.num_mlp_layer}")
    print(f"remove_one_hop: {cfg.model.remove_one_hop}")
    print(f"learning_rate: {cfg.optimizer.lr}")
    print(f"{separator}\n")
    
    # Fixed
    cfg.model.aggregate_func = "pna"
    cfg.model.use_pyg_propagation = True
    cfg.model.randomized_edge_drop = 0.0
    
    # Architecture choices - keep fixed to match pretrained model
    cfg.model.short_cut = random.choice([True, False])
    cfg.model.dependent = random.choice([True, False])
    # cfg.model.concat_hidden = random.choice([True, False])
    cfg.model.remove_one_hop = random.choice([True, False])
    
    # Task Parameters
    cfg.task.num_negative = random.choice([16, 32, 64])
    cfg.task.strict_negative = random.choice([True, False])
    cfg.task.adversarial_temperature = random.choice([0.5, 1.0, 1.5, 2.0])
    
    # Optimizer Parameters
    cfg.optimizer.lr = random.choice([1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3])
    
    # Training Parameters
    cfg.train.batch_size = random.choice([4, 8, 16])
    
    # Add run ID to wandb name for tracking
    cfg.wandb.name = cfg.wandb.name + f"_{args.hyper_run_id}"
    
    # Create timestamped directory structure
    dataset_class = cfg.dataset["class"].lower()
    
    # Use a shared timestamp file to ensure all runs use the same directory
    timestamp_file = f"hyperparams/.current_timestamp_{dataset_class}"
    
    if args.hyper_run_id == 0:
        # First run: create new timestamp and directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        hyperparams_dir = f"hyperparams/{timestamp}_{dataset_class}_finetune"
        os.makedirs(hyperparams_dir, exist_ok=True)
        
        # Save timestamp for other runs to use
        with open(timestamp_file, 'w') as f:
            f.write(hyperparams_dir)
        print(f"Created new hyperparameter search directory: {hyperparams_dir}")
    else:
        # Subsequent runs: use existing timestamp
        if os.path.exists(timestamp_file):
            with open(timestamp_file, 'r') as f:
                hyperparams_dir = f.read().strip()
        else:
            # Fallback if timestamp file doesn't exist
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            hyperparams_dir = f"hyperparams/{timestamp}_{dataset_class}_finetune"
            os.makedirs(hyperparams_dir, exist_ok=True)
    
    # Save the configuration with dataset-specific name in timestamped directory
    config_filename = f"{hyperparams_dir}/hyperparam_cfg_nbfnet_{dataset_class}.pt"
    torch.save(cfg, config_filename)
    
    # Create/append to consolidated CSV file
    csv_filename = f"{hyperparams_dir}/hyperparameter_combinations.csv"
    csv_headers = [
        "run_id", "timestamp", "input_dim", "hidden_dims", 
        "message_func", "aggregate_func", "batch_size", "learning_rate", 
        "num_negative", "short_cut", "layer_norm", "dependent", 
        "concat_hidden", "num_mlp_layer", "remove_one_hop", 
        "strict_negative", "adversarial_temperature"
    ]
    
    # Check if CSV file exists to determine if we need headers
    csv_exists = os.path.exists(csv_filename)
    
    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write headers only if file doesn't exist
        if not csv_exists:
            writer.writerow(csv_headers)
        
        # Write the current hyperparameter combination
        writer.writerow([
            args.hyper_run_id,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            cfg.model.input_dim,
            str(cfg.model.hidden_dims),
            cfg.model.message_func,
            cfg.model.aggregate_func,
            cfg.train.batch_size,
            cfg.optimizer.lr,
            cfg.task.num_negative,
            cfg.model.short_cut,
            cfg.model.layer_norm,
            cfg.model.dependent,
            cfg.model.concat_hidden,
            cfg.model.num_mlp_layer,
            cfg.model.remove_one_hop,
            cfg.task.strict_negative,
            cfg.task.adversarial_temperature
        ])
    
    # Store the hyperparams directory path for later use by training script
    cfg.hyperparams_dir = hyperparams_dir
    
    print(f"Configuration saved to: {config_filename}")
    print(f"Hyperparameters logged to: {csv_filename}")
