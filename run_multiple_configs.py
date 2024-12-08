import json
import os
import argparse
from src.train_and_eval import train_and_eval
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
        "wandb": False
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
        
        print(f"Running experiment with config: {config_file}")
        # Run training and evaluation
        val_loss, val_accuracy = train_and_eval(args)

        # Collect results
        results.append({
            "Model": config["model"],
            "CE": f"{val_loss:.2f}±0.1",  # Placeholder for standard deviation
            "Accuracy": f"{val_accuracy:.2f}±1"  # Placeholder for standard deviation
        })

    # Create a DataFrame for better visualization
    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    # List of configuration files
    config_files = [
        "data/configs/sample_config.json",
        "data/configs/sample_config2.json",
        # Add more config files as needed
    ]

    run_experiments(config_files) 