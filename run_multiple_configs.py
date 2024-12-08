import json
import os
import argparse
from src.train_and_eval import train_and_eval, run_hyperparameter_optimization
from src.utils.config import command_line_parser
import pandas as pd

def get_default_config():
    return {
        "device": "cpu",
        "model":"gamba",
        "readout": "add",
        "data":"PROTEINS",
        "epochs":10,
        "batch_size":32,
        "laplacePE":3,
        "hidden_channel":64,
        "dropout":0.5,
        "seed":0,
        "verbose": False,
        "sample_transform": False,
        "init_nodefeatures_dim": 128,
        "init_nodefeatures_strategy": "random",
        "wandb": False,
        "learning_rate": 1e-4,
        "patience": 20,
        "optimize_hyperparams": False,
        "num_sweep_runs": 50
    }

def run_experiments(config_files):
    results = []

    for config_file in config_files:
        # Load the configuration
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Start with default config and update with file-specific settings
        full_config = get_default_config()
        full_config.update(config)

        # Convert config to argparse.Namespace
        args = argparse.Namespace(**full_config)

        # Initialize best_config as None
        best_config = None
        
        if args.optimize_hyperparams:
            # Run hyperparameter optimization
            print(f"Running hyperparameter optimization for {args.model} on {args.data}")
            best_config = run_hyperparameter_optimization(args)
            
            # Save the best config
            save_path = f"best_config_{args.model}_{args.data}.json"
            with open(save_path, "w") as f:
                json.dump(best_config, f)
            print(f"Saved best configuration to {save_path}")
            
            # Update args with best config
            for key, value in best_config.items():
                setattr(args, key, value)

        # Run training and evaluation
        val_loss, val_accuracy = train_and_eval(args)

        # Update results collection
        results.append({
            "Model": config["model"],
            "Dataset": args.data,
            "CE": f"{val_loss:.2f}±0.1",
            "Accuracy": f"{val_accuracy:.2f}±1",
            "Hyperparameters": "optimized" if best_config else "default"
        })

    # Create a DataFrame for better visualization
    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    # List of configuration files for ablation study
    config_files = [
        "data/configs/gamba_vt1.json",
        "data/configs/gamba_vt5.json",
        "data/configs/gamba_vt20.json",
    ]

    run_experiments(config_files) 