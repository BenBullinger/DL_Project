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

@timer
def train_and_eval(args):
    if args.verbose:
        print("Running with the following arguments:")
        print(json.dumps(args.__dict__, indent=2))

    if args.wandb:
        wandb.init(
            project="DL_Project",
            config=args.__dict__
        )
        
        # add Git hash to the run
        try:
            git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
            wandb.config.git_hash = git_hash
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
            normalization="layernorm",
            dropout=args.dropout,
            use_enc=True,
            use_dec=True,
            use_readout=args.readout if task_info["task_type"] == "graph_prediction" else None,
            num_virtual_tokens=args.num_virtual_tokens
        ).to(device)
    
    return train(model, train_loader, val_loader, args)

def train(model, train_loader, val_loader, args):
    """Training function with optional W&B logging"""
    device = args.device
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    best_val_accuracy = 0
    patience = args.patience if hasattr(args, 'patience') else 20
    patience_counter = 0
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    for epoch in tqdm(range(0, args.epochs+1), desc="Training", unit="epoch"):
        model.train()
        total_loss = 0
        correct = 0
        total_samples = 0

        for batch in train_loader:
            batch.to(device)
            optimizer.zero_grad()

            edge_attr = getattr(batch, 'edge_attr', None)
            output = model(batch.x, batch.edge_index, batch.batch, 
                          edge_attr=edge_attr, laplacePE=(None if not hasattr(batch, "laplacePE") else batch.laplacePE))
            
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

        if not epoch % 20 and val_loader is not None:
            val_loss, val_accuracy = evaluate(model, val_loader, args)
            
            # Only log to wandb if it's enabled
            if args.wandb:
                wandb.log({
                    "train_loss": avg_loss, 
                    "train_accuracy": avg_accuracy,
                    "val_loss": val_loss, 
                    "val_accuracy": val_accuracy,
                    "epoch": epoch
                })
            
            tqdm.write(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}, Train Acc: {avg_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if val_loader is not None:
        final_val_loss, final_val_accuracy = evaluate(model, val_loader, args)
        # Log final metrics only if wandb is enabled
        if args.wandb:
            wandb.log({
                "final_val_loss": final_val_loss,
                "final_val_accuracy": final_val_accuracy
            })
        return final_val_loss, final_val_accuracy

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
                          edge_attr=edge_attr, laplacePE=(None if not hasattr(batch, "laplacePE") else batch.laplacePE))
            
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

if __name__ == "__main__":
    # Only execute if run directly, not when imported
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()
    train_and_eval(args)