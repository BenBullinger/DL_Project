from torch_geometric.nn import summary
from torch_geometric.data import Data
from src.nn.gamba_ARSP_multi import GambaARSP
from src.nn.gamba_simple import Gamba
from src.nn.gat_super import GATSuper
import torch

hidden_channels = 64
num_virtual_tokens = 100

# Example synthetic graph data
num_nodes = 500
num_edges = 20
num_node_features = hidden_channels
num_edge_features = hidden_channels

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create synthetic data
x = torch.randn(num_nodes, num_node_features, device=device)  # Node features
edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)  # Edge indices
edge_attr = torch.randn(num_edges, num_edge_features, device=device)  # Edge features
batch = torch.zeros(num_nodes, dtype=torch.long, device=device) 

# Your custom PyG model
model = GambaARSP(in_channels=hidden_channels, hidden_channels=hidden_channels, out_channels=2, layers=1, mlp_depth=2, num_virtual_tokens=num_virtual_tokens, use_enc=True, use_dec=True).to(device)
model_simple = Gamba(
                in_channels=hidden_channels,
                hidden_channels=hidden_channels,
                layers=1,
                out_channels=hidden_channels,
                mlp_depth=2,
                num_virtual_tokens=num_virtual_tokens,
                normalization="layernorm",
                use_enc=True,
                use_dec=True,
            ).to(device)
model_gat = GATSuper(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            layers=1,
            out_channels=2,
            heads=1,
            normalization="layernorm",
            dropout=0,
            use_enc=True,
            use_dec=True,
            ).to(device)    
# PyG summary
print(summary(model_gat, x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch))