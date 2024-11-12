import torch
import torch.nn as nn
from torch_geometric.nn import GINConv
from transformers import MambaConfig, MambaModel

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

class GraphAttentionAggregator(nn.Module):
    """Aggregates graph information into a fixed number of nodes using transformer attention"""
    def __init__(self, hidden_channels, num_virtual_tokens, num_attention_heads=4):
        super().__init__()
        self.num_virtual_tokens = num_virtual_tokens
        
        # Virtual tokens that will learn to attend to the graph
        self.virtual_tokens = nn.Parameter(torch.randn(num_virtual_tokens, hidden_channels))
        
        # Use transformer's multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=num_attention_heads,
            batch_first=True
        )
        
    def forward(self, x, batch):
        # Expand virtual tokens for batch size
        batch_size = torch.max(batch) + 1
        virtual_tokens = self.virtual_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Reshape x to [batch_size, num_nodes_per_graph, channels]
        x_batched = torch.zeros(batch_size, max(batch.bincount()), x.size(-1), device=x.device)
        for i in range(batch_size):
            batch_mask = batch == i
            x_batched[i, :batch_mask.sum()] = x[batch_mask]
        
        # Apply attention
        out, _ = self.attention(
            query=virtual_tokens,
            key=x_batched,
            value=x_batched
        )
        
        return out  # [batch_size, num_virtual_tokens, channels]

class Gamba(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        layers,
        out_channels,
        mlp_depth=2,
        normalization="layernorm",
        dropout=0.5,
        use_enc=True,
        use_dec=True,
        use_readout="add",
        num_virtual_tokens=4,
        num_attention_heads=4
    ):
        super().__init__()
        
        self.num_layers = layers
        
        # Initial GIN layer
        self.input_gin = GINConv(nn.Linear(in_channels, hidden_channels))
        
        # Configure Mamba
        self.mamba_config = MambaConfig(
            hidden_size=hidden_channels,
            intermediate_size=hidden_channels * 2,
            num_hidden_layers=1
        )
        self.mamba = MambaModel(self.mamba_config)
        
        # Main layers
        self.layers = nn.ModuleList()
        for _ in range(layers):
            layer = nn.ModuleDict({
                'attention': GraphAttentionAggregator(
                    hidden_channels=hidden_channels,
                    num_virtual_tokens=num_virtual_tokens,
                    num_attention_heads=num_attention_heads
                ),
                'gin': GINConv(nn.Linear(hidden_channels, hidden_channels))
            })
            self.layers.append(layer)
            
        # Output layers
        self.output_gin = GINConv(nn.Linear(hidden_channels, out_channels))
        
        # Add readout if specified
        self.readout = None
        if use_readout:
            supported_pools = {'add': global_add_pool, 'mean': global_mean_pool, 'max': global_max_pool}
            self.readout = supported_pools.get(use_readout, global_add_pool)
        
    def forward(self, x, edge_index, batch, **kwargs):
        # Initial GIN layer
        x = self.input_gin(x, edge_index)
        
        # Main layers
        for layer in self.layers:
            # Regular message passing
            gin_out = layer['gin'](x, edge_index)
            
            # Global information processing
            global_tokens = layer['attention'](x, batch)
            global_features = self.mamba(inputs_embeds=global_tokens).last_hidden_state
            global_update = global_features.mean(dim=1)
            
            # Add global information to each node's features
            x = gin_out + global_update[batch]
            
        # Final GIN layer
        x = self.output_gin(x, edge_index)
        
        # Apply readout if specified
        if self.readout is not None:
            x = self.readout(x, batch)
        
        return x