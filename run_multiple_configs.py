import json
import os
import argparse
from src.train_and_eval import train_and_eval
from src.utils.config import command_line_parser
import pandas as pd
import wandb

def get_default_config():
    return {
        "device": "cpu",
        "model":"gamba",
        "readout": "add",
        "data":"PROTEINS",
        "epochs":10,
        "batch_size":32,
        "laplacePE":0,
        "RW_length":0,
        "hidden_channel":64,
        "layers":4,
        "heads":4,
        "dropout":0.2,
        "seed":0,
        "verbose": False,
        "sample_transform": False,
        "add_virtual_node":True,
        "init_nodefeatures_dim": 128,
        "init_nodefeatures_strategy": "random",
        "wandb": False,
        "weight_decay": 0.01,
        "learning_rate": 1e-4,
        "scheduler": "None",
        "scheduler_patience": 16,
        "simon_gaa": False,
        "num_virtual_tokens": 4,
        "token_aggregation": "mean",
        "use_mamba": True,
        "patience": 20
    }

def run_experiments(config_files):
    results = []

    # Initialize wandb once before the loop
    if get_default_config()["wandb"]:
        wandb.init(
            project="DL_Project",
            config=args.__dict__,
            name=args.name
        )

    for config_file in config_files:
        # Load the configuration
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Start with default config and update with file-specific settings
        full_config = get_default_config()
        full_config.update(config)

        # Convert config to argparse.Namespace
        args = argparse.Namespace(**full_config)

        # Run training and evaluation
        val_loss, val_accuracy = train_and_eval(args)

        # Collect results
        results.append({
            "Model": config["name"],
            "Loss": f"{val_loss:.4f}",
            "Accuracy": f"{val_accuracy:.4f}"
        })

    # Create a DataFrame for better visualization
    df = pd.DataFrame(results)
    print("\nResults:")
    print(df.to_markdown(index=False))

    # Close wandb run
    if get_default_config()["wandb"]:
        wandb.finish()

if __name__ == "__main__":
    # List of configuration files
    config_files = [
        "data/configs/archived_configs/6h_run.json",
        #"data/configs/sample_config2.json",
        #"data/configs/sample_config3.json",
        #"data/configs/sample_config4.json"
        # Add more config files as needed
    ]

    run_experiments(config_files) 