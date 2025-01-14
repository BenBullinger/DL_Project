import os
import sys

# Add project root to Python path when running directly
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, project_root)

import numpy as np
import json
import torch
from tqdm import tqdm
from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
from torch_geometric.graphgym.models.encoder import AtomEncoder, BondEncoder
import torch.nn.functional as F
from src.nn.gin import GIN
from src.nn.gat_super import GATSuper
from src.nn.graph_transformer import GraphTransformerNet
from src.nn.gamba_simple import Gamba
from src.nn.gamba_simple_multi import GambaMulti
from src.nn.gamba import GambaAR
from src.nn.gamba_PVOC_multi import GambaSP
from src.nn.gamba_ARSP_multi import GambaARSP
from src.utils.preprocess import preprocess_dataset, explicit_preprocess, fix_splits
from src.utils.dataset import load_data
from src.utils.misc import seed_everything, timer
import src.utils.metrics as metrics
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
        if(hasattr(args, "name")):
            naming = args.name
        else:
            naming = f"{args.model}_{args.data}"

        wandb.init(
            entity="astinky",
            project="DL_Project",
            config=args.__dict__,
            name=naming
        )
        
        # add Git hash to the run
        try:
            git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
            wandb.config.update(args.__dict__, allow_val_change=True)
            print(f"Logged Git hash: {git_hash}")
        except subprocess.CalledProcessError as e:
            print("Error retrieving Git hash. Make sure you are in a Git repository.", e)

    seed_everything(args.seed)

    device = args.device
    
    train_loader, val_loader, test_loader, task_info = load_data(args)
    atom_encoder = AtomEncoder(emb_dim=args.hidden_channel).to(device) if task_info["needs_ogb_encoder"] else None
    bond_encoder = BondEncoder(emb_dim=args.hidden_channel).to(device) if task_info["needs_ogb_encoder"] else None
    task_info["node_feature_dims"] = args.hidden_channel if task_info["needs_ogb_encoder"] else task_info["node_feature_dims"]
    task_info["edge_feature_dims"] = args.hidden_channel if task_info["needs_ogb_encoder"] else task_info["edge_feature_dims"]

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
    elif args.model == "gat":
        model = GATSuper(
            in_channels=task_info["node_feature_dims"],
            hidden_channels=args.hidden_channel,
            layers=1,
            out_channels=task_info["output_dims"],
            heads=args.heads,
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
        if args.layers > 1:
            model = GambaMulti(
                in_channels=task_info["node_feature_dims"],
                hidden_channels=args.hidden_channel,
                layers=args.layers,
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
        else:            
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
    elif args.model == "gambaAR":
            model = GambaAR(
                in_channels=task_info["node_feature_dims"],
                hidden_channels=args.hidden_channel,
                layers=args.layers,
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
    elif args.model == "gambaSP":
            model = GambaSP(
                in_channels=task_info["node_feature_dims"],
                hidden_channels=args.hidden_channel,
                layers=args.layers,
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
    elif args.model == "gambaARSP":
            model = GambaARSP(
                in_channels=task_info["node_feature_dims"],
                hidden_channels=args.hidden_channel,
                layers=args.layers,
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
    return train(model, train_loader, val_loader, test_loader, atom_encoder=atom_encoder, bond_encoder=bond_encoder, args=args)


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

def train(model, train_loader, val_loader, test_loader, atom_encoder, bond_encoder, args, config=None):
    """Modified training function to support hyperparameter tuning"""
    device = args.device
    
    # Use config parameters if provided (for hyperparameter tuning)
    if config is not None:
        for param, value in config.items():
            setattr(args, param, value)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler_dict =  {
        'None': torch.optim.lr_scheduler.LambdaLR(optimizer, (lambda x : 1), last_epoch=- 1, verbose=False),
        'Plateau': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.8, patience=args.scheduler_patience, threshold=0.1, threshold_mode='rel', min_lr=1e-6),
    }
    scheduler = scheduler_dict[args.scheduler]

    best_val_accuracy = 0
    patience = args.patience if hasattr(args, 'patience') else 20
    patience_counter = 0
    
    loss_fn = metrics.train_losses_dict[args.data][0]
    loss_fn_name = metrics.train_losses_dict[args.data][1]
    report_metric = metrics.report_metric_dict[args.data]
    
    for epoch in tqdm(range(0, args.epochs+1), desc="Training", unit="epoch"):
        model.train()
        total_loss = 0
        correct = 0
        total_samples = 0

        for batch in train_loader:
            batch.to(device)
            optimizer.zero_grad()

            if atom_encoder is not None:
                batch = atom_encoder(batch)
            if bond_encoder is not None:
                batch = bond_encoder(batch)
            edge_attr = getattr(batch, 'edge_attr', None)
            output = model(batch.x, batch.edge_index, batch.batch,
                          edge_attr=edge_attr, laplacePE=(None if not hasattr(batch, "laplacePE") else batch.laplacePE), rwse=(None if not hasattr(batch, "random_walk_pe") else batch.random_walk_pe))
           
            if output.dim() == 1:
                output = output.unsqueeze(0)
            if loss_fn_name == "BCE":
                output = torch.sigmoid(output)
            loss = loss_fn(output, batch.y)
            loss.backward()
            optimizer.step()

            predictions = output.argmax(dim=-1)
            if loss_fn_name in ["BCE", "MSE"]:
                avg_accuracy = float('NaN')
            else:
                correct += (predictions == batch.y).sum().item()
                total_samples += batch.y.size(0)
                avg_accuracy = correct / total_samples
            
            total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)

        if val_loader is not None:
            val_loss, val_report = evaluate(model, val_loader, args, atom_encoder=atom_encoder, bond_encoder=bond_encoder)
            scheduler_param_dict =  {
            'None': None,
            'Plateau': val_loss,
            }
            scheduler.step(scheduler_param_dict[args.scheduler])


        if not epoch % 20 and val_loader is not None:
            train_loss, train_report = evaluate(model, train_loader, args, atom_encoder=atom_encoder, bond_encoder=bond_encoder)
            val_loss, val_report = evaluate(model, val_loader, args, atom_encoder=atom_encoder, bond_encoder=bond_encoder)
            test_loss, test_report = evaluate(model, test_loader, args, atom_encoder=atom_encoder, bond_encoder=bond_encoder)
            if args.wandb:
                wandb.log({
                    "train_loss": train_loss, 
                    f"train_{report_metric}": train_report,
                    "val_loss": val_loss, 
                    f"val_{report_metric}": val_report,
                    "test_loss": test_loss,
                    f"test_{report_metric}": test_report,
                    "epoch": epoch
                })
            tqdm.write(f"Epoch {epoch} - Train Loss ({loss_fn_name}): {train_loss:.4f}, Train {report_metric}: {train_report:.4f}, Val Loss ({loss_fn_name}): {val_loss:.4f}, Val {report_metric}: {val_report:.4f}, Test Loss ({loss_fn_name}): {test_loss:.4f}, Test {report_metric}: {test_report:.4f}, 'lr': {optimizer.param_groups[0]['lr']:.8f}")

            if val_report > best_val_accuracy:
                best_val_accuracy = val_report
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            # Report metrics to Ray Tune if we're doing hyperparameter optimization
            if config is not None:
                tune.report(
                    val_accuracy=val_report,
                    val_loss=val_loss,
                    train_accuracy=avg_accuracy,
                    train_loss=avg_loss,
                    epoch=epoch
                )
    wandb.finish()
    if val_loader is not None:
        return evaluate(model, val_loader, args, atom_encoder=atom_encoder, bond_encoder=bond_encoder)

def evaluate(model, val_loader, args, atom_encoder, bond_encoder):
    report_metric = metrics.report_metric_dict[args.data]
    if report_metric == "Acc":
        return metrics.evaluate_acc(model, val_loader, args, atom_encoder, bond_encoder)
    if report_metric == "F1":
        return metrics.evaluate_f1(model, val_loader, args, atom_encoder, bond_encoder)
    if report_metric == "AP":
        return metrics.evaluate_ap(model, val_loader, args, atom_encoder, bond_encoder)
    if report_metric == "MAE":
        return metrics.evaluate_mae(model, val_loader, args, atom_encoder, bond_encoder)
    else:
        raise ValueError(f"\"{report_metric}\" is either not a valid metric, or not implemented.")

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
    import argparse
    parser = argparse.ArgumentParser()
    
    # Add all arguments with their default values
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on")
    parser.add_argument("--model", type=str, default="gamba", help="Model type (gamba/gin/gt)")
    parser.add_argument("--readout", type=str, default="add", help="Readout function")
    parser.add_argument("--data", type=str, default="PROTEINS", help="Dataset name")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--laplacePE", type=int, default=3, help="Laplacian PE dimension")
    parser.add_argument("--hidden_channel", type=int, default=64, help="Hidden channel dimension")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--sample_transform", action="store_true", help="Use sample transform")
    parser.add_argument("--init_nodefeatures_dim", type=int, default=128, help="Initial node features dimension")
    parser.add_argument("--init_nodefeatures_strategy", type=str, default="random", help="Strategy for initial node features")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--optimize_hyperparams", action="store_true", help="Run hyperparameter optimization")
    parser.add_argument("--num_virtual_tokens", type=int, default=10, help="Number of virtual tokens for Gamba model")
    parser.add_argument("--scheduler", type=str, default="None", choices=["None", "Plateau"],
                       help="Learning rate scheduler (None or Plateau)")
    parser.add_argument("--simon_gaa", action="store_true", help="Use Simon GAA attention mechanism")

    args = parser.parse_args()
    main(args)