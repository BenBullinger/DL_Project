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
        print("Loading Weights & Biases configuration")
        if(hasattr(args, "name")):
            naming = args.name
        else:
            naming = f"{args.model}_{args.data}"
        print(f"Run name: {naming}")
        wandb.init(
            project="DL_Project",
            config={
                "model": args.model,
                "seed": args.seed,
                "epochs": args.epochs,
                "data": args.data,
                "batch_size": args.batch_size,
                "hidden_channels": args.hidden_channel,
                "dropout": args.dropout,
                "laplacePE": args.laplacePE,
                "init_nodefeatures_dim": args.init_nodefeatures_dim,
                "init_nodefeatures_strategy": args.init_nodefeatures_strategy,
                "readout": args.readout,
                # "ignore_GNNBenchmark_original_split": args.ignore_GNNBenchmark_original_split,
            },
            name=naming
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
            use_readout=args.readout if task_info["task_type"] == "graph_prediction" else None
        ).to(device)
    
    return train(model, train_loader, val_loader, args=args)


def train(model, train_loader, val_loader, args, **kwargs):
    device = args.device
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
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
            if args.wandb:
                wandb.log({
                    "train_loss": avg_loss, 
                    "train_accuracy": avg_accuracy,
                    "val_loss": val_loss, 
                    "val_accuracy": val_accuracy,
                    "epoch": epoch
                })
            tqdm.write(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}, Train Acc: {avg_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
    wandb.finish()

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