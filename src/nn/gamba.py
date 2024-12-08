import torch
import torch.nn as nn
from torch_scatter import scatter_mean
from torch_geometric.nn import GINConv
from transformers import MambaConfig, MambaModel

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

class TokenAggregator(nn.Module):
    """Aggregate graph information into a fixed set of tokens."""
    def __init__(self):
        super().__init__()

    def forward(self, x, batch=None):
        if batch is not None:
            return scatter_mean(x, batch, dim=0)
        else:
            return x.mean(dim=1)

class GraphAttentionAggregator(nn.Module):
    """Aggregates graph information into a fixed number of nodes using transformer attention"""
    def __init__(self, hidden_channels, num_virtual_tokens, args=None, num_attention_heads=4):
        super().__init__()
        self.num_virtual_tokens = num_virtual_tokens
        self.aggr = TokenAggregator()
        
        # Virtual tokens that will learn to attend to the graph
        self.virtual_tokens = nn.Parameter(torch.randn(num_virtual_tokens, hidden_channels))
        
        # Use transformer's multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=num_attention_heads,
            batch_first=True
        )
        
    def forward(self, x, batch):
        batch_size = torch.max(batch) + 1

        virtual_tokens = self.virtual_tokens.unsqueeze(0).expand(batch_size, -1, -1).clone()
        #virtual_tokens.shape = [batch, num_virtual_tokens, hidden_channel]
        aggregated_tokens = self.aggr(x, batch)
        aggregated_tokens = aggregated_tokens.unsqueeze(1) # [batch_size, channels]
        
        x_batched = torch.zeros(batch_size, max(batch.bincount()), x.size(-1), device=x.device)
        #input(f"Virtual tokens at start:\n{virtual_tokens[0,:,:]}")
        for i in range(self.virtual_tokens.size(0)):
            query = aggregated_tokens
            out, _ = self.attention(
                query=query,
                key=x_batched,
                value=x_batched
            )
            aggregated_tokens = self.aggr(torch.cat((virtual_tokens[:,0:i,:], out), dim=1)).unsqueeze(1)
            virtual_tokens[:,i,:] = aggregated_tokens.squeeze(1)
            #input(f"Virtual tokens after iteration {i}:\n{virtual_tokens[0,:,:]}")
        
        return virtual_tokens  # [batch_size, num_virtual_tokens, channels]

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
        num_attention_heads=4,
        args=None
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
        
        # Add layer normalization
        self.layer_norm = nn.LayerNorm(hidden_channels)
        
    def forward(self, x, edge_index, batch, **kwargs):
        x = self.input_gin(x, edge_index)
        
        for layer in self.layers:
            # Regular message passing
            gin_out = layer['gin'](x, edge_index)
            
            # Global information processing
            global_tokens = layer['attention'](x, batch)
            global_features = self.mamba(inputs_embeds=global_tokens).last_hidden_state
            global_update = global_features.mean(dim=1)
            
            # Normalized feature combination
            x = self.layer_norm(gin_out + 0.1 * global_update[batch])  # Scale global contribution
            
        x = self.output_gin(x, edge_index)
        
        # Apply readout if specified
        if self.readout is not None:
            x = self.readout(x, batch)
        
        return x