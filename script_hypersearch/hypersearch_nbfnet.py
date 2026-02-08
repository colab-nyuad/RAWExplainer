import os
import sys
import random
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nbfnet import util

separator = ">" * 30
line = "-" * 30

if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    
    # Model Architecture Parameters - keep input_dim fixed to avoid compatibility issues
    # Only vary the hidden layer dimensions, not input/output layers
    
    # Number of hidden layers (between input and output)
    num_layers = random.choice([4, 5, 6])
    
    # Hidden layer dimension - only the middle layers, not input/output
    hidden_dim = random.choice([16, 32, 64, 128])
    
    # Create hidden_dims list: [input_dim, hidden_dim, hidden_dim, ..., hidden_dim]
    # Keep input_dim fixed from config, vary only the intermediate hidden dimensions
    base_input_dim = cfg.model.input_dim  # Use the fixed input_dim from config
    cfg.model.hidden_dims = [base_input_dim] + [hidden_dim] * (num_layers - 1)
    
    # Fixed
    cfg.model.message_func = "distmult"
    
    # Log the hyperparameters being used
    print(f"\n{separator}")
    print(f"HYPERPARAMETER COMBINATION:")
    print(f"base_input_dim: {base_input_dim} (fixed from config)")
    print(f"hidden_dims: {cfg.model.hidden_dims}")
    print(f"hidden_layer_dim: {hidden_dim} (varied)")
    print(f"message_func: {cfg.model.message_func} (fixed)")
    print(f"total_layers: {num_layers}")
    print(f"{separator}\n")
    
    # Fixed
    cfg.model.aggregate_func = "pna"
    cfg.model.use_pyg_propagation = True
    cfg.model.randomized_edge_drop = 0.0
    
    # Architecture choices
    cfg.model.short_cut = random.choice([True, False])
    cfg.model.layer_norm = random.choice([True, False])
    cfg.model.dependent = random.choice([True, False])
    cfg.model.concat_hidden = random.choice([True, False])
    cfg.model.num_mlp_layer = random.choice([1, 2, 3, 4])
    cfg.model.remove_one_hop = random.choice([True, False])
    
    # Task Parameters - reduced for memory efficiency
    cfg.task.num_negative = random.choice([16, 32, 64])
    cfg.task.strict_negative = random.choice([True, False])
    cfg.task.adversarial_temperature = random.choice([0.5, 1.0, 1.5, 2.0])
    
    # Optimizer Parameters
    cfg.optimizer.lr = random.choice([1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
    
    # Training Parameters - reduced batch sizes for memory efficiency
    cfg.train.batch_size = random.choice([4, 8, 16])
    
    # Add run ID to wandb name for tracking
    cfg.wandb.name = cfg.wandb.name + f"_{args.hyper_run_id}"
    
    # Create hyperparams directory if it doesn't exist
    os.makedirs("hyperparams", exist_ok=True)
    
    # Save the configuration with dataset-specific name
    dataset_class = cfg.dataset["class"].lower()
    config_filename = f"hyperparams/hyperparam_cfg_nbfnet_{dataset_class}.pt"
    torch.save(cfg, config_filename)
    
    # Also save a human-readable version for debugging
    debug_filename = f"hyperparams/hyperparam_debug_{dataset_class}_{args.hyper_run_id}.txt"
    with open(debug_filename, 'w') as f:
        f.write(f"Hyperparameter Configuration - Run ID: {args.hyper_run_id}\n")
        f.write(f"{'='*50}\n")
        f.write(f"input_dim: {cfg.model.input_dim}\n")
        f.write(f"hidden_dims: {cfg.model.hidden_dims}\n")
        f.write(f"message_func: {cfg.model.message_func}\n")
        f.write(f"aggregate_func: {cfg.model.aggregate_func}\n")
        f.write(f"num_layers: {num_layers}\n")
        f.write(f"batch_size: {cfg.train.batch_size}\n")
        f.write(f"learning_rate: {cfg.optimizer.lr}\n")
        f.write(f"num_negative: {cfg.task.num_negative}\n")
        f.write(f"short_cut: {cfg.model.short_cut}\n")
        f.write(f"layer_norm: {cfg.model.layer_norm}\n")
        f.write(f"dependent: {cfg.model.dependent}\n")
        f.write(f"concat_hidden: {cfg.model.concat_hidden}\n")
        f.write(f"num_mlp_layer: {cfg.model.num_mlp_layer}\n")
        f.write(f"remove_one_hop: {cfg.model.remove_one_hop}\n")
        f.write(f"strict_negative: {cfg.task.strict_negative}\n")
        f.write(f"adversarial_temperature: {cfg.task.adversarial_temperature}\n")
    
    print(f"Configuration saved to: {config_filename}")
    print(f"Debug info saved to: {debug_filename}")
