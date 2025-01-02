import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, GatedGraphConv
from transformers import MambaConfig, MambaModel
from torch_geometric.utils import to_dense_batch

from .mlp import get_mlp
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

class GambaMulti(nn.Module):
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
        
        self.input_gin = GINConv(get_mlp(input_dim=first_gin_dim, hidden_dim=hidden_channels, mlp_depth=mlp_depth, output_dim=hidden_channels, normalization=torch.nn.LayerNorm, last_relu=False), train_eps=True)
        
        self.pe_gnn = GatedGraphConv(out_channels=hidden_channels, num_layers=8)

        self.theta_layers = nn.ModuleList([
            nn.Linear(hidden_channels*2, num_virtual_tokens, bias=False)
            for _ in range(layers)
        ])
        
        # Configure Mamba
        self.mamba_configs = [
            MambaConfig(
                hidden_size=hidden_channels*2,
                intermediate_size=hidden_channels * 2,
                num_hidden_layers=4
            )
            for _ in range(layers)
        ]
        self.mamba_layers = nn.ModuleList([MambaModel(config) for config in self.mamba_configs])
 
        self.merge_layers = nn.ModuleList([
            get_mlp(
                input_dim=hidden_channels * 3, hidden_dim=hidden_channels,
                output_dim=hidden_channels, mlp_depth=1,
                normalization=torch.nn.LayerNorm, last_relu=False
            )
            for _ in range(layers)
        ])

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
        x_orig =x
        pe = self.pe_gnn(x, edge_index)
        #x = self.input_gin(x, edge_index)
        
        for i in range(self.num_layers):

            x = torch.cat([x, pe], dim=1)
            
            x_dense, mask = to_dense_batch(x, batch)
            alpha = self.theta_layers[i](x_dense).transpose(1,2)
            alpha_X = alpha @ x_dense

            x_mamba = self.mamba_layers[i](inputs_embeds=alpha_X).last_hidden_state
        
            x_m = x_mamba[batch]
            x = self.merge_layers[i](torch.cat([x_orig, x_m[:,-1,:]], dim=1)) 
        
        x = self.output_gin(x, edge_index)
        
        # Apply readout if specified
        if self.readout is not None:
            x = self.readout(x, batch)

        if self.dec is not None:
            x = self.dec(x)
        
        return x