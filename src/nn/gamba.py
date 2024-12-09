import torch
import torch.nn as nn
from torch_scatter import scatter_mean
from torch_geometric.nn import GINConv
from transformers import MambaConfig, MambaModel

from .mlp import get_mlp

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

class TokenAggregator(nn.Module):
    """Aggregate graph information into a fixed set of tokens."""
    def __init__(self, hidden_channels, strategy="mean"):
        super().__init__()
        self.strategy = strategy
        if strategy == "gru":
            self.rnn = nn.GRU(hidden_channels, hidden_channels, batch_first=True)
        
    def forward(self, x, batch=None):
        
        if self.strategy == "mean":
            if batch is not None:
                return scatter_mean(x, batch, dim=0)
            else:
                #input(x.mean(dim=1).shape)
                return x.mean(dim=1)
            
        if self.strategy == "gru":
            if batch is not None:
                return scatter_mean(x, batch, dim=0)
            else:
                batch_size = x.shape[0]
                out, hidden_state = self.rnn(x)
                hidden_state = hidden_state.permute(1, 0, 2).reshape(batch_size, -1) 
                return hidden_state.squeeze(1)

class GraphAttentionAggregator(nn.Module):
    """Aggregates graph information into a fixed number of nodes using transformer attention"""
    def __init__(self, hidden_channels, num_virtual_tokens, args=None, num_attention_heads=4):
        super().__init__()
        self.num_virtual_tokens = num_virtual_tokens
        self.args = args
        self.aggr = TokenAggregator(hidden_channels, strategy=args.token_aggregation)
        
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
        if self.args.simon_gaa:
            out, _ = self.attention(
            query=virtual_tokens,
            key=x_batched,
            value=x_batched
            )

            return out
         
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
        self.args = args
        # Initial GIN layer
        if use_enc:
            self.enc = get_mlp(input_dim=in_channels, hidden_dim=hidden_channels, mlp_depth=mlp_depth, output_dim=hidden_channels, normalization=torch.nn.LayerNorm, last_relu=False)
        first_gin_dim = hidden_channels if use_enc else in_channels
        self.input_gin = GINConv(nn.Linear(first_gin_dim, hidden_channels), train_eps=True)
        
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
                    num_attention_heads=num_attention_heads,
                    args=args
                ),
                'gin': GINConv(nn.Linear(hidden_channels, hidden_channels))
            })
            self.layers.append(layer)
            
        # Output layers
        last_gin_dim = hidden_channels if use_dec else out_channels
        self.output_gin = GINConv(nn.Linear(hidden_channels, last_gin_dim))
        if use_dec:
            self.dec = get_mlp(input_dim=hidden_channels, hidden_dim=hidden_channels, mlp_depth=mlp_depth, output_dim=out_channels, normalization=torch.nn.LayerNorm, last_relu=False)
        
        # Add readout if specified
        self.readout = None
        if use_readout:
            supported_pools = {'add': global_add_pool, 'mean': global_mean_pool, 'max': global_max_pool}
            self.readout = supported_pools.get(use_readout, global_add_pool)
        
        # Add layer normalization
        self.layer_norm = nn.LayerNorm(hidden_channels)
        
    def forward(self, x, edge_index, batch, **kwargs):
        if self.enc is not None:
            x = self.enc(x)
        x = self.input_gin(x, edge_index)
        
        for layer in self.layers:
            # Regular message passing
            gin_out = layer['gin'](x, edge_index)
            
            # Global information processing
            global_tokens = layer['attention'](x, batch)
            if self.args.use_mamba:
                global_features = self.mamba(inputs_embeds=global_tokens).last_hidden_state
            else:
                raise NotImplementedError("For now you have to use mamba")
            global_update = global_features.mean(dim=1)
            
            # Normalized feature combination
            x = self.layer_norm(gin_out + 0.1 * global_update[batch])  # Scale global contribution
            
        x = self.output_gin(x, edge_index)
        
        # Apply readout if specified
        if self.readout is not None:
            x = self.readout(x, batch)

        if self.dec is not None:
            x = self.dec(x)
        
        return x