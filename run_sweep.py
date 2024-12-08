import wandb
import subprocess
import argparse
from pathlib import Path
import yaml

def load_wandb_key():
    """Load Weights & Biases API key from wandb.key file"""
    try:
        with open('wandb.key', 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError("wandb.key file not found. Please create one with your W&B API key.")

def run_sweep(sweep_config_path: str, count: int = 50):
    """
    Initialize and run a W&B sweep
    
    Args:
        sweep_config_path: Path to the sweep configuration YAML file
        count: Number of runs to execute in the sweep
    """
    # Set up W&B API key
    wandb.login(key=load_wandb_key())
    
    # Load sweep configuration
    with open(sweep_config_path, 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    # Initialize the sweep
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project="DL_Project"
    )
    
    # Start the sweep agent
    wandb.agent(sweep_id, count=count)
    
    return sweep_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_config", type=str, default="sweeps/gamba_sweep.yaml",
                      help="Path to sweep configuration file")
    parser.add_argument("--count", type=int, default=50,
                      help="Number of runs to execute in the sweep")
    
    args = parser.parse_args()
    
    # Ensure the sweeps directory exists
    Path("sweeps").mkdir(exist_ok=True)
    
    try:
        # Run the sweep
        sweep_id = run_sweep(args.sweep_config, args.count)
        print(f"Sweep started with ID: {sweep_id}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please create a wandb.key file with your W&B API key.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()  # This will print the full error traceback 