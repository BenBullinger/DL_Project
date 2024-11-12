import json
import torch
from tqdm import tqdm
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch.utils.data import random_split
from src.nn.gin import GIN
from src.nn.graph_transformer import GraphTransformerNet
from src.utils.preprocess import preprocess_dataset, explicit_preprocess
from src.nn.gamba import Gamba

def train_and_eval(args):
    if args.verbose:
        print("Running with the following arguments:")
        print(json.dumps(args.__dict__, indent=2))
    
    device = args.device
    dataset_name = args.data.upper()
    
    ########################
    """
    TODO: The following has yet to be implemented
    1. 
        Check what dataset name is specified and either fetch it from TUDataset, or from GNNBenchmarkDataset
        (https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.GNNBenchmarkDataset.html#torch_geometric.datasets.GNNBenchmarkDataset)
        Given the dataset_name is CIFAR10, ideally you initialise it as
        dataset = GNNBenchmarkDataset(root=data/GGNBenchmarkDataset, name='CIFAR10')

    2. 
        Come up with a logic where we can use pre_transform to store the prepocessed graphs instead of preprocessing them all the time
    3. 
        Fix not only seeds, but also ensure that it is always the same set of train/val/test dataset instead of sampling randomly
        (probably we can set the seed to 0 first, then sample the dataset and then set the seed to some other value if needed)
    """
    if dataset_name in ["REDDIT", "IMDB"]:
        dataset_name += "-BINARY"
    dataset = TUDataset(root="data/TUDataset", name=dataset_name, use_node_attr=True)
    #######################
    total_size = len(dataset)
    datalist_prepocessed = explicit_preprocess(datalist=list(dataset), transform=preprocess_dataset(args))
    train_size = int(0.8 * total_size) 
    val_size = int(0.1 * total_size)    
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(datalist_prepocessed, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    if args.model == "gin":
        model = GIN(
            in_channels=datalist_prepocessed[0].x.shape[1],
            hidden_channels=args.hidden_channel,
            layers=1,
            out_channels=dataset.num_classes,
            mlp_depth=2,
            normalization="layernorm",
            dropout=args.dropout,
            use_enc=True,
            use_dec=True,
            use_readout="add"
        ).to(device)
    elif args.model == "gt":
        model = GraphTransformerNet(
            node_dim_in=datalist_prepocessed[0].x.shape[1],
            edge_dim_in=dataset.num_edge_features,
            out_dim=dataset.num_classes,
            pe_in_dim=args.laplacePE,
            hidden_dim=args.hidden_channel,
            num_heads=8,
            dropout=args.dropout
        ).to(device)
    elif args.model == "gamba":
        model = Gamba(
            in_channels=datalist_prepocessed[0].x.shape[1],
            hidden_channels=args.hidden_channel,
            layers=1,
            out_channels=dataset.num_classes,
            mlp_depth=2,
            normalization="layernorm",
            dropout=args.dropout,
            use_enc=True,
            use_dec=True,
            use_readout="add"
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

    print("Training completed.")

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