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
from src.utils.misc import seed_everything
from src.nn.gamba import Gamba

def train_and_eval(args):
    if args.verbose:
        print("Running with the following arguments:")
        print(json.dumps(args.__dict__, indent=2))

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
    
    train(model, train_loader, val_loader, args=args)


def train(model, train_loader, val_loader, args, **kwargs):
    device = args.device
    # Set up optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    # Training loop
    model.train()
    for epoch in tqdm(range(0, args.epochs+1), desc="Training", unit="epoch"):
        total_loss = 0
        correct = 0
        total_samples = 0

        for batch in train_loader:
            batch.to(device)
            optimizer.zero_grad()

            # Get edge_attr from batch if it exists, otherwise None
            edge_attr = getattr(batch, 'edge_attr', None)
            output = model(batch.x, batch.edge_index, batch.batch, 
                          edge_attr=edge_attr, laplacePE=batch.laplacePE)
            
            # Compute loss
            loss = loss_fn(output, batch.y)
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            predictions = output.argmax(dim=1)
            correct += (predictions == batch.y).sum().item()
            total_samples += batch.y.size(0)

            total_loss += loss.item()


        # Display validation metrics
        if not epoch % 20 and val_loader is not None:
            val_loss, val_accuracy = evaluate(model, val_loader, args)
            tqdm.write(f"Validation - Epoch {epoch}, Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

def evaluate(model, val_loader, args):
    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0
    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in val_loader:
            # Move batch to the device
            batch.to(args.device)
            
            # Get edge_attr from batch if it exists, otherwise None
            edge_attr = getattr(batch, 'edge_attr', None)
            output = model(batch.x, batch.edge_index, batch.batch,
                          edge_attr=edge_attr, laplacePE=batch.laplacePE)
            
            # Compute loss
            loss = loss_fn(output, batch.y)
            total_loss += loss.item()

            # Calculate accuracy
            predictions = output.argmax(dim=1)
            correct += (predictions == batch.y).sum().item()
            total_samples += batch.y.size(0)

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total_samples
    model.train()  # Switch back to training mode
    return avg_loss, accuracy