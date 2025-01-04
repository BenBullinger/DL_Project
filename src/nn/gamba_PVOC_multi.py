import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, MessagePassing
from transformers import MambaConfig, MambaModel
from torch_geometric.utils import to_dense_batch
import torch.nn.functional as F

from .mlp import get_mlp
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

class SpatialConv(MessagePassing):
    def __init__(self, hidden_channels):
        super().__init__(aggr='add')
        # Separate projections for edge features
        self.sobel_proj = nn.Linear(1, hidden_channels)  # Sobel filter values
        self.boundary_proj = nn.Linear(1, hidden_channels)  # Boundary counts
        
        self.node_mlp = get_mlp(
            input_dim=hidden_channels * 2,  # node_i + node_j
            hidden_dim=hidden_channels,
            output_dim=hidden_channels,
            mlp_depth=1,
            normalization=torch.nn.LayerNorm,
            last_relu=False
        )
        
        self.final_norm = nn.LayerNorm(hidden_channels)
        
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
    def message(self, x_i, x_j, edge_attr):
        # Split edge features
        sobel_values = edge_attr[:, 0:1]  # Sobel filter values
        boundary_counts = edge_attr[:, 1:2]  # Boundary pixel counts
        
        # Process node features
        node_features = self.node_mlp(torch.cat([x_i, x_j], dim=-1))
        
        # Process edge features
        sobel_weights = self.sobel_proj(sobel_values)  # Edge intensity
        boundary_weights = self.boundary_proj(boundary_counts)  # Boundary strength
        
        # Combine all information
        # Higher boundary count and Sobel values indicate stronger connections
        edge_importance = torch.sigmoid(sobel_weights + boundary_weights)
        
        return node_features * edge_importance

    def update(self, aggr_out, x):
        # Residual connection and normalization
        return self.final_norm(x + aggr_out)

class MultiScaleGambaLayer(nn.Module):
    def __init__(self, hidden_channels, num_virtual_tokens):
        super().__init__()
        # Local spatial processing at different scales
        self.spatial_convs = nn.ModuleList([
            SpatialConv(hidden_channels) for _ in range(3)  # 3 scales
        ])
        
        # Global processing with Mamba
        self.theta = nn.Linear(hidden_channels, num_virtual_tokens, bias=False)
        self.mamba_config = MambaConfig(
            hidden_size=hidden_channels,
            intermediate_size=hidden_channels * 2,
            num_hidden_layers=1
        )
        self.mamba = MambaModel(self.mamba_config)
        
        # Merge multi-scale features
        self.merge = get_mlp(
            input_dim=hidden_channels * 4,  # 3 scales + mamba
            hidden_dim=hidden_channels,
            output_dim=hidden_channels,
            mlp_depth=1,
            normalization=torch.nn.LayerNorm,
            last_relu=False
        )
        
        self.layer_norm = nn.LayerNorm(hidden_channels)
        
    def forward(self, x, edge_index, edge_attr, batch):
        identity = x
        
        # Multi-scale spatial processing
        xs = []
        for conv in self.spatial_convs:
            xs.append(conv(x, edge_index, edge_attr))
        
        # Global processing
        x_dense, mask = to_dense_batch(x, batch)
        alpha = self.theta(x_dense).transpose(1,2)
        alpha_X = alpha @ x_dense
        x_mamba = self.mamba(inputs_embeds=alpha_X).last_hidden_state
        x_m = x_mamba[batch][:,-1,:]  # Use last token
        
        # Merge all features
        x = self.merge(torch.cat([*xs, x_m], dim=-1))
        
        # Residual connection and normalization
        x = self.layer_norm(x + identity)
        
        return x

class GambaSP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        layers=3,  # Reduced from 5 to 3 for efficiency
        mlp_depth=2,
        normalization="layernorm",
        dropout=0.5,
        use_enc=True,
        use_dec=True,
        use_readout="add",
        num_virtual_tokens=4,
        num_attention_heads=4,
        args=None
    ):
        super().__init__()
        
        print("Gamba PVOC Multi-Scale with", layers, "layers")
        
        self.num_layers = layers
        self.args = args
        
        # Initial encoding
        if use_enc:
            self.enc = get_mlp(
                input_dim=in_channels, 
                hidden_dim=hidden_channels, 
                mlp_depth=mlp_depth, 
                output_dim=hidden_channels, 
                normalization=torch.nn.LayerNorm, 
                last_relu=False
            )
        
        # Multi-scale Gamba layers
        self.layers = nn.ModuleList([
            MultiScaleGambaLayer(hidden_channels, num_virtual_tokens)
            for _ in range(layers)
        ])
        
        # Output layers
        if use_dec:
            self.dec = get_mlp(
                input_dim=hidden_channels, 
                hidden_dim=hidden_channels, 
                mlp_depth=mlp_depth, 
                output_dim=out_channels, 
                normalization=torch.nn.LayerNorm, 
                last_relu=False
            )
        
        self.readout = None
        if use_readout:
            supported_pools = {'add': global_add_pool, 'mean': global_mean_pool, 'max': global_max_pool}
            self.readout = supported_pools.get(use_readout, global_add_pool)
        
    def forward(self, x, edge_index, batch, edge_attr=None, **kwargs):
        if self.enc is not None:
            x = self.enc(x)
        
        # Apply multi-scale layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, batch)
        
        if self.readout is not None:
            x = self.readout(x, batch)

        if self.dec is not None:
            x = self.dec(x)
        
        return x