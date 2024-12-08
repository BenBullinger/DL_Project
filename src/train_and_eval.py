import numpy as np
import json
import torch
from tqdm import tqdm
from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from src.nn.gin import GIN
from src.nn.graph_transformer import GraphTransformerNet
from src.utils.preprocess import preprocess_dataset, explicit_preprocess, fix_splits
from src.utils.dataset import load_data
from src.utils.misc import seed_everything, timer
from src.nn.gamba import Gamba
import wandb
import subprocess
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

@timer
def train_and_eval(args):
    if args.verbose:
        print("Running with the following arguments:")
        print(json.dumps(args.__dict__, indent=2))

    if args.wandb:
        wandb.config.update(args.__dict__, allow_val_change=True)
        
        # add Git hash to the run
        try:
            git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
            wandb.config.update({"git_hash": git_hash}, allow_val_change=True)
            print(f"Logged Git hash: {git_hash}")
        except subprocess.CalledProcessError as e:
            print("Error retrieving Git hash. Make sure you are in a Git repository.", e)

    seed_everything(args.seed)

    device = args.device
    dataset_name = args.data.upper()
    
    train_loader, val_loader, test_loader, task_info = load_data(args)
    
    if args.model == "gin":
        model = GIN(
            in_channels=task_info["node_feature_dims"],
            hidden_channels=args.hidden_channel,
            layers=1,
            out_channels=task_info["output_dims"],
            mlp_depth=2,
            normalization="layernorm",
            dropout=args.dropout,
            use_enc=True,
            use_dec=True,
            use_readout=args.readout if task_info["task_type"] == "graph_prediction" else None
        ).to(device)
    elif args.model == "gt":
        model = GraphTransformerNet(
            node_dim_in=task_info["node_feature_dims"],
            edge_dim_in=task_info["edge_feature_dims"],
            out_dim=task_info["output_dims"],
            pe_in_dim=args.laplacePE,
            hidden_dim=args.hidden_channel,
            num_heads=8,
            dropout=args.dropout
        ).to(device)
    elif args.model == "gamba":
        model = Gamba(
            in_channels=task_info["node_feature_dims"],
            hidden_channels=args.hidden_channel,
            layers=1,
            out_channels=task_info["output_dims"],
            mlp_depth=2,
            num_virtual_tokens=args.num_virtual_tokens,
            normalization="layernorm",
            dropout=args.dropout,
            use_enc=True,
            use_dec=True,
            args=args,
            use_readout=args.readout if task_info["task_type"] == "graph_prediction" else None
        ).to(device)
    
    return train(model, train_loader, val_loader, args=args)


def get_hyperparameter_space(model_name):
    """Define hyperparameter search space for each model"""
    if model_name == "gin":
        return {
            "hidden_channels": tune.choice([32, 64, 128, 256]),
            "dropout": tune.uniform(0.0, 0.5),
            "learning_rate": tune.loguniform(1e-4, 1e-2),
            "layers": tune.choice([1, 2, 3, 4]),
            "mlp_depth": tune.choice([1, 2, 3])
        }
    elif model_name == "gt":
        return {
            "hidden_channels": tune.choice([32, 64, 128, 256]),
            "dropout": tune.uniform(0.0, 0.5),
            "learning_rate": tune.loguniform(1e-4, 1e-2),
            "num_heads": tune.choice([4, 8, 16])
        }
    elif model_name == "gamba":
        return {
            "hidden_channels": tune.choice([32, 64, 128, 256]),
            "dropout": tune.uniform(0.0, 0.5),
            "learning_rate": tune.loguniform(1e-4, 1e-2),
            "layers": tune.choice([1, 2, 3, 4]),
            "mlp_depth": tune.choice([1, 2, 3])
        }

def train(model, train_loader, val_loader, args, config=None):
    """Modified training function to support hyperparameter tuning"""
    device = args.device
    
    # Use config parameters if provided (for hyperparameter tuning)
    if config is not None:
        for param, value in config.items():
            setattr(args, param, value)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler_dict =  {
        'None': torch.optim.lr_scheduler.LambdaLR(optimizer, (lambda x : 1), last_epoch=- 1, verbose=False),
        'Plateau': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.7, patience=12, threshold=0.1, threshold_mode='rel', min_lr=0.00001),
    }
    scheduler = scheduler_dict[args.scheduler]

    best_val_accuracy = 0
    patience = args.patience if hasattr(args, 'patience') else 20
    patience_counter = 0
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    for epoch in tqdm(range(0, args.epochs+1), desc="Training", unit="epoch"):
        model.train()
        total_loss = 0
        correct = 0
        total_samples = 0

        # Print first batch predictions
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx == 0:  # Only for first batch
                print("\nFirst 5 samples of training batch:")
                print(f"Labels:      {batch.y[:20].cpu().numpy()}")
                
                batch.to(device)
                edge_attr = getattr(batch, 'edge_attr', None)
                output = model(batch.x, batch.edge_index, batch.batch,
                             edge_attr=edge_attr, laplacePE=(None if not hasattr(batch, "laplacePE") else batch.laplacePE))
                
                predictions = output.argmax(dim=-1)
                print(f"Predictions: {predictions[:20].cpu().numpy()}")
                print(f"Raw outputs:\n{output[:20].cpu().detach().numpy()}\n")
            
            # Regular training loop continues...
            batch.to(device)
            optimizer.zero_grad()

            edge_attr = getattr(batch, 'edge_attr', None)
            output = model(batch.x, batch.edge_index, batch.batch,
                          edge_attr=edge_attr, laplacePE=(None if not hasattr(batch, "laplacePE") else batch.laplacePE) )
            
            if output.dim() == 1:
                output = output.unsqueeze(0)

            
            
            loss = loss_fn(output, batch.y)
            loss.backward()
            optimizer.step()

            predictions = output.argmax(dim=-1)
            correct += (predictions == batch.y).sum().item()
            total_samples += batch.y.size(0)
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        avg_accuracy = correct / total_samples

        if val_loader is not None:
            val_loss, val_accuracy = evaluate(model, val_loader, args)
            scheduler_param_dict =  {
            'None': None,
            'Plateau': val_loss,
            }
            scheduler.step(scheduler_param_dict[args.scheduler])


        if not epoch % 20 and val_loader is not None:
            val_loss, val_accuracy = evaluate(model, val_loader, args)
            if args.wandb:
                wandb.log({
                    "train_loss": avg_loss, 
                    "train_accuracy": avg_accuracy,
                    "val_loss": val_loss, 
                    "val_accuracy": val_accuracy,
                    "epoch": epoch
                })
            tqdm.write(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}, Train Acc: {avg_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, 'lr': {optimizer.param_groups[0]['lr']}")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            # Report metrics to Ray Tune if we're doing hyperparameter optimization
            if config is not None:
                tune.report(
                    val_accuracy=val_accuracy,
                    val_loss=val_loss,
                    train_accuracy=avg_accuracy,
                    train_loss=avg_loss,
                    epoch=epoch
                )

    if val_loader is not None:
        return evaluate(model, val_loader, args)

def evaluate(model, val_loader, args):
    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0
    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in val_loader:
            batch.to(args.device)
            
            edge_attr = getattr(batch, 'edge_attr', None)
            output = model(batch.x, batch.edge_index, batch.batch,
                          edge_attr=edge_attr, laplacePE=(None if not hasattr(batch, "laplacePE") else batch.laplacePE) )
            
            if output.dim() == 1:
                output = output.unsqueeze(0)
            
            loss = loss_fn(output, batch.y)
            total_loss += loss.item()

            predictions = output.argmax(dim=-1)
            correct += (predictions == batch.y).sum().item()
            total_samples += batch.y.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total_samples
    
    return avg_loss, accuracy

def run_hyperparameter_optimization(args):
    """Run hyperparameter optimization using Ray Tune"""
    search_space = get_hyperparameter_space(args.model)
    
    scheduler = ASHAScheduler(
        max_t=args.epochs,
        grace_period=10,
        reduction_factor=2
    )
    
    search_algo = OptunaSearch()
    
    analysis = tune.run(
        tune.with_parameters(train_and_eval, args=args),
        config=search_space,
        num_samples=50,  # number of trials
        scheduler=scheduler,
        search_alg=search_algo,
        resources_per_trial={"cpu": 2, "gpu": 0.5},  # adjust based on your hardware
        metric="val_accuracy",
        mode="max",
        name=f"hpo_{args.model}_{args.data}"
    )
    
    best_trial = analysis.get_best_trial("val_accuracy", "max", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['val_accuracy']}")
    
    return best_trial.config

def main(args):
    if args.optimize_hyperparams:
        best_config = run_hyperparameter_optimization(args)
        # Save best config for future use
        with open(f"best_config_{args.model}_{args.data}.json", "w") as f:
            json.dump(best_config, f)
    else:
        train_and_eval(args)

if __name__ == "__main__":
    # Only execute if run directly, not when imported
    import argparse
    parser = argparse.ArgumentParser()
    # Add your arguments here
    parser.add_argument("--optimize_hyperparams", action="store_true", help="Run hyperparameter optimization")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()
    main(args)