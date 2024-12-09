"""
Loads the datasets.
"""
import torch
from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
from utils.preprocess import preprocess_dataset, explicit_preprocess, fix_splits

def load_data(args):
    task_description = {}
    if args.verbose:
        print(f"Loading {args.data}")
    dataset_name = args.data.upper()
    if dataset_name in ["CIFAR10", "MNIST", "CLUSTER", "PATTERN", "TSP"]:
        train_loader, val_loader, test_loader, info = GNNBenchmarkLoader(args, dataset_name)
    if dataset_name in ["MUTAG", "ENZYMES", "PROTEINS", "COLLAB", "IMDB", "REDDIT"]:
        train_loader, val_loader, test_loader, info = TUDatasetLoader(args, dataset_name)
    return train_loader, val_loader, test_loader, info

def GNNBenchmarkLoader(args, dataset_name):
    train_dataset = GNNBenchmarkDataset(root="data/GNNBenchmarkDataset", name=dataset_name, split="train", pre_transform=preprocess_dataset(args))
    val_dataset = GNNBenchmarkDataset(root="data/GNNBenchmarkDataset", name=dataset_name, split="val" , pre_transform=preprocess_dataset(args))
    test_dataset = GNNBenchmarkDataset(root="data/GNNBenchmarkDataset", name=dataset_name, split="test" , pre_transform=preprocess_dataset(args))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    info = create_info(train_dataset)
    
    return train_loader, val_loader, test_loader, info

def TUDatasetLoader(args, dataset_name):
    if dataset_name in ["REDDIT", "IMDB"]:
        dataset_name += "-BINARY"
    dataset = TUDataset(root="data/TUDataset", name=dataset_name, use_node_attr=True)
    datalist_prepocessed = explicit_preprocess(datalist=list(dataset), transform=preprocess_dataset(args))
    train_dataset, val_dataset, test_dataset = fix_splits(dataset=datalist_prepocessed, ratio=(0.8,0.1,0.1), shuffle=True)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)   
    info = create_info(dataset, datalist_prepocessed[0])

    return train_loader, val_loader, test_loader, info

def create_info(dataset, sample=None):
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

    return {
        "node_feature_dims": node_feature_dims,
        "output_dims": num_classes,
        "edge_feature_dims": edge_feature_dims,
        "task_type": task_type,
    }