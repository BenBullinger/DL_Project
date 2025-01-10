import torch
from torch import Tensor
import torch_geometric.typing
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.transforms import BaseTransform

import random
from tqdm import tqdm
from typing import Any, Optional

def add_node_attr(
    data: Data,
    value: Any,
    attr_name: Optional[str] = None,
) -> Data:

    if attr_name is None:
        if data.x is not None:
            x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
        else:
            data.x = value
    else:
        data[attr_name] = value

    return data

def preprocess_dataset(config):
    """
    Wrapper function to apply your transforms
    """
    transforms = []
    if config.sample_transform:
        transforms.append(SampleTransform())
    transforms.append(InitEmptyNodeFeatures(dimension=config.init_nodefeatures_dim, strategy=config.init_nodefeatures_strategy))
    if config.add_virtual_node:
        transforms.append(T.VirtualNode())
    if config.laplacePE > 0:
        transforms.append(T.AddLaplacianEigenvectorPE(k=config.laplacePE, attr_name="laplacePE"))
        if config.model == "gin":
            transforms.append(ConcatToNodeFeatures(attr_name="laplacePE"))
    if config.RW_length > 0:
        transforms.append(T.AddRandomWalkPE(walk_length=config.RW_length))

    return T.Compose(transforms)

class SampleTransform(BaseTransform):
    """
    This is just an example transform, in case you want to write a custom one
    """
    def __init__(self):
        pass

    def forward(self, data: Data) -> Data:
        """
        Here, you can write your transform on a single graph, it is then applied on a full dataset
        """
        data = add_node_attr(data, "You run sample transform", attr_name="sample_check")

class ConcatToNodeFeatures(BaseTransform):
    def __init__(self,
                 attr_name: Optional[str]):
        self.attr_name = attr_name

    def forward(self, data:Data) -> Data:
        assert data.x is not None, "If you see this error, Robert failed debugging the node feature init properly (tell him pls)"
        data.x = torch.cat((data.x, data[self.attr_name]), dim=1)
        return data
    
class InitEmptyNodeFeatures(BaseTransform):
    def __init__(self,
                 dimension: int,
                 strategy: str="ones"):
        self.k = dimension
        self.strategy = strategy

    def forward(self, data:Data) -> Data:
        if data.x is None: #We are only doing this, if there are no node features
            if self.strategy == "random":
                # Initialize node features with random values
                data.x = torch.rand(data.num_nodes, self.k, dtype=torch.float, device=data.edge_index.device)
            if self.strategy == "ones":
                # Initialize node features with ones
                data.x = torch.ones(data.num_nodes, self.k, dtype=torch.float, device=data.edge_index.device)
            if self.strategy == "zeroes":
                # Initialize node features with ones
                data.x = torch.zeros(data.num_nodes, self.k, dtype=torch.float, device=data.edge_index.device)
        return data

def explicit_preprocess(datalist, transform):
    for idx in tqdm(range(len(datalist)), desc="Preprocessing the dataset"):
        datalist[idx] = transform(datalist[idx])
    return datalist

def fix_splits(dataset, ratio:tuple=(0.8, 0.1, 0.1), shuffle=False):
    """
    Creates train, val and test split.
    
    Args:
        dataset: The dataset to split
        ratio: Ratio of train:val:test split
        shuffle: Randomly shuffles the dataset.
    """
    assert sum(ratio) == 1, f"Ratio ({ratio}) should sum up to 1"
    assert ratio[0] > 0, f"Training set needs at least one sample"

    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)
    train_split = int(ratio[0] * len(indices))
    val_split = int((ratio[0]+ratio[1]) * len(indices))

    if isinstance(dataset, list):
        train_dataset = [dataset[i] for i in indices[:train_split]]
        val_dataset = [dataset[i] for i in indices[train_split:val_split]]
        test_dataset = [dataset[i] for i in indices[val_split:]]

        return train_dataset, val_dataset, test_dataset

    elif isinstance(dataset, DataLoader):
        dataset.train_idx = indices[:train_split]
        dataset.val_idx = indices[train_split:val_split]
        dataset.test_idx = indices[val_split:]

        return dataset[indices[:train_split]], dataset[indices[:val_split]], dataset[indices[val_split:]]
    
    else:
        raise Exception(f"The dataset which you want to split is not supported (is of type {type(dataset)})")