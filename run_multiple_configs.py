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
        "regularization":"simple",
        "readout": "add",
        "data":"PROTEINS",
        "epochs":10,
        "batch_size":32,
        "pe":"gnn",
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

def run_experiments(config_files, num_trials=3):
    results = []

    # Initialize wandb once before the loop
    if get_default_config()["wandb"]:
        wandb.init(
            entity="astinky",
            project="DL_Project",
            config=args.__dict__,
            name=args.name
        )

    for config_file in config_files:
        # Load the configuration
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Run multiple trials for each config
        for trial in range(num_trials):
            # Start with default config and update with file-specific settings
            full_config = get_default_config()
            full_config.update(config)
            
            # Update seed for each trial to ensure different random initializations
            full_config["seed"] = trial

            # Convert config to argparse.Namespace
            args = argparse.Namespace(**full_config)

            # Run training and evaluation
            val_loss, val_accuracy = train_and_eval(args)

            # Collect results
            results.append({
                "Model": config["name"],
                "Trial": trial + 1,
                "Loss": float(val_loss),
                "Accuracy": float(val_accuracy)
            })

    # Create a DataFrame for better visualization
    df = pd.DataFrame(results)
    
    # Calculate statistics per model
    stats = df.groupby('Model').agg({
        'Loss': ['mean', 'std'],
        'Accuracy': ['mean', 'std']
    }).round(4)
    
    print("\nResults per run:")
    print(df.to_markdown(index=False))
    
    print("\nStatistics per model:")
    print(stats.to_markdown(floatfmt='.4f'))

    # Log results to wandb
    if get_default_config()["wandb"]:
        # Log individual runs
        for _, row in df.iterrows():
            wandb.log({
                "individual_runs/model": row["Model"],
                "individual_runs/trial": row["Trial"],
                "individual_runs/loss": row["Loss"],
                "individual_runs/accuracy": row["Accuracy"]
            })
        
        # Log aggregate statistics
        for model in stats.index:
            wandb.log({
                "aggregate_stats/model": model,
                "aggregate_stats/loss_mean": stats.loc[model, ('Loss', 'mean')],
                "aggregate_stats/loss_std": stats.loc[model, ('Loss', 'std')],
                "aggregate_stats/accuracy_mean": stats.loc[model, ('Accuracy', 'mean')],
                "aggregate_stats/accuracy_std": stats.loc[model, ('Accuracy', 'std')]
            })

        # Create and log a wandb Table with all results
        wandb_table = wandb.Table(dataframe=df)
        wandb.log({"results_table": wandb_table})

        wandb.finish()

if __name__ == "__main__":
    # List of configuration files
    config_files = [
        "data/configs/archived_configs/stats_test.json",
        #"data/configs/sample_config2.json",
        #"data/configs/sample_config.json",
        #"data/configs/sample_config2.json",
        #"data/configs/sample_config3.json",
        #"data/configs/sample_config4.json"
    ]

    run_experiments(config_files, num_trials=3) 