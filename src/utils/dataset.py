"""
Loads the datasets.
"""
import torch
from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset, LRGBDataset
from torch_geometric.loader import DataLoader
from src.utils.preprocess import preprocess_dataset, explicit_preprocess, fix_splits
import os
from pathlib import Path

def get_project_root():
    """Get the project root directory from any file in the project"""
    # Start from the current file's directory
    current_dir = Path(__file__).resolve().parent
    # Go up until we find the project root (where data/ exists)
    while current_dir.name != "DL_Project" and current_dir.parent != current_dir:
        current_dir = current_dir.parent
    return current_dir

def load_data(args):
    #Sorry for this mess of a code but LRGBDataset doesn't capitalize their dataset names smh
    
    if args.verbose:
        print(f"Loading {args.data}")
    dataset_name = args.data
    if dataset_name.upper() in ["CIFAR10", "MNIST", "CLUSTER", "PATTERN", "TSP"]:
        train_loader, val_loader, test_loader, info = GNNBenchmarkLoader(args, dataset_name.upper())
    elif dataset_name.upper() in ["MUTAG", "ENZYMES", "PROTEINS", "COLLAB", "IMDB", "REDDIT"]:
        train_loader, val_loader, test_loader, info = TUDatasetLoader(args, dataset_name.upper())
    elif dataset_name in ["PascalVOC-SP", "COCO-SP", "PCQM-Contact", "Peptides-func", "Peptides-struct"]:
        train_loader, val_loader, test_loader, info = LRGBDatasetLoader(args, dataset_name)
    else:
        raise ValueError(f"Dataset called {dataset_name} was not found. Check out the available dataset options once we enable that")
    return train_loader, val_loader, test_loader, info

def GNNBenchmarkLoader(args, dataset_name):
    data_dir = os.path.join(get_project_root(), "data", "GNNBenchmarkDataset")
    train_dataset = GNNBenchmarkDataset(root=data_dir, name=dataset_name, split="train", pre_transform=preprocess_dataset(args))
    val_dataset = GNNBenchmarkDataset(root=data_dir, name=dataset_name, split="val" , pre_transform=preprocess_dataset(args))
    test_dataset = GNNBenchmarkDataset(root=data_dir, name=dataset_name, split="test" , pre_transform=preprocess_dataset(args))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    info = create_info(train_dataset, args)
    
    return train_loader, val_loader, test_loader, info

def TUDatasetLoader(args, dataset_name):
    if dataset_name in ["REDDIT", "IMDB"]:
        dataset_name += "-BINARY"
    
    data_dir = os.path.join(get_project_root(), "data", "TUDataset")
    dataset = TUDataset(
        root=data_dir,
        name=dataset_name, 
        use_node_attr=True
    )
    datalist_prepocessed = explicit_preprocess(datalist=list(dataset), transform=preprocess_dataset(args))
    train_dataset, val_dataset, test_dataset = fix_splits(dataset=datalist_prepocessed, ratio=(0.8,0.1,0.1), shuffle=True)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)   
    info = create_info(dataset, args, datalist_prepocessed[0])

    return train_loader, val_loader, test_loader, info

def LRGBDatasetLoader(args, dataset_name):
    data_dir = os.path.join(get_project_root(), "data", "LRGBDataset")
    train_dataset = LRGBDataset(root=data_dir, name=dataset_name, split="train", pre_transform=preprocess_dataset(args))
    val_dataset = LRGBDataset(root=data_dir, name=dataset_name, split="val" , pre_transform=preprocess_dataset(args))
    test_dataset = LRGBDataset(root=data_dir, name=dataset_name, split="test" , pre_transform=preprocess_dataset(args))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)  
    info = create_info(train_dataset, args)

    return train_loader, val_loader, test_loader, info


def create_info(dataset, args, sample=None):
    """
    Given a dataset, it takes a sample and creates the task overview dictionary.
    Also determines whether the task is graph-level or node-level prediction.
    If a sample is passed, input dimensions are computed based on that sample instead of taking one out of the dataset
    """
    num_classes = dataset.num_classes
    if sample is None:
        sample = dataset[0]
    node_feature_dims = sample.x.shape[1]
    edge_feature_dims = sample.num_edge_features if hasattr(sample, 'num_edge_features') else None

    # Check if it's a batch (multiple graphs) or a single graph
    if sample.batch is not None: #Is batched
        num_graphs = sample.batch.max().item() + 1
        if sample.y.shape[0] == num_graphs:
            task_type = "graph_prediction"
        elif sample.y.shape[0] == sample.x.shape[0]: 
            task_type = "node_prediction"
        else:
            raise ValueError(f"Cannot determine task type for batch with y shape {sample.y.shape}")
    else:  # Single graph
        if sample.y.shape[0] == 1:
            task_type = "graph_prediction"
        elif sample.y.shape[0] == sample.x.shape[0]:
            task_type = "node_prediction"
        else:
            raise ValueError(f"Cannot determine task type for graph with y shape {sample.y.shape}")
    
    needs_ogb_encoder = args.data in ["Peptides-func", "Peptides-struct"]        

    return {
        "node_feature_dims": node_feature_dims,
        "output_dims": num_classes,
        "edge_feature_dims": edge_feature_dims,
        "task_type": task_type,
        "needs_ogb_encoder" :needs_ogb_encoder
    }