import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, GatedGraphConv
from transformers import MambaConfig, MambaModel
from torch_geometric.utils import to_dense_batch

from .mlp import get_mlp
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

class GambaAR(nn.Module):
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
        self.virtual_tokens = num_virtual_tokens
        self.hidden_channels = hidden_channels

        # Initial GIN layer
        if use_enc:
            self.enc = get_mlp(
                input_dim=in_channels, hidden_dim=hidden_channels,
                mlp_depth=mlp_depth, output_dim=hidden_channels,
                normalization=torch.nn.LayerNorm, last_relu=False
            )
        first_gin_dim = hidden_channels if use_enc else in_channels
        
        self.input_gin = GINConv(
            get_mlp(
                input_dim=first_gin_dim, hidden_dim=hidden_channels,
                mlp_depth=mlp_depth, output_dim=hidden_channels,
                normalization=torch.nn.LayerNorm, last_relu=False
            ),
            train_eps=True
        )
        
        self.pe_gnn = GatedGraphConv(out_channels=hidden_channels, num_layers=8)

        # GRU cell and weights for virtual token updates
        self.gru_cell = nn.GRUCell(hidden_channels * 2, hidden_channels)
        self.weights = get_mlp(
            input_dim=hidden_channels * 3, hidden_dim=hidden_channels * 3,
            mlp_depth=mlp_depth, output_dim=hidden_channels * 2,
            normalization=torch.nn.LayerNorm, last_relu=False
        )
        
        # Configure Mamba
        self.mamba_config = MambaConfig(
            hidden_size=hidden_channels * 2,
            intermediate_size=hidden_channels * 2,
            num_hidden_layers=4
        )
        self.mamba = MambaModel(self.mamba_config)
        self.layer_norm_mamba = nn.LayerNorm(hidden_channels * 2)
        
        self.merge = get_mlp(
            input_dim=hidden_channels * 3, hidden_dim=hidden_channels,
            output_dim=hidden_channels, mlp_depth=1,
            normalization=torch.nn.LayerNorm, last_relu=False
        )
            
        # Output layers
        last_gin_dim = hidden_channels if use_dec else out_channels
        self.output_gin = GINConv(nn.Linear(hidden_channels, last_gin_dim))
        if use_dec:
            self.dec = get_mlp(
                input_dim=hidden_channels, hidden_dim=hidden_channels,
                mlp_depth=mlp_depth, output_dim=out_channels,
                normalization=torch.nn.LayerNorm, last_relu=False
            )
        
        # Add readout if specified
        self.readout = None
        if use_readout:
            supported_pools = {'add': global_add_pool, 'mean': global_mean_pool, 'max': global_max_pool}
            self.readout = supported_pools.get(use_readout, global_add_pool)
        
    def forward(self, x, edge_index, batch, **kwargs):
        if self.enc is not None:
            x = self.enc(x)
        x_orig = x
        pe = self.pe_gnn(x, edge_index)

        x_dense, mask = to_dense_batch(torch.cat([x, pe], dim=1), batch)

        ts = []
        h = torch.zeros(x_dense.shape[0], x_orig.shape[-1]).to(device=x.device)
        for _ in range(self.virtual_tokens):
            alpha = self.weights(torch.cat([x_dense, h.unsqueeze(1).expand(-1, x_dense.shape[1], -1)], dim=-1))
            t = torch.einsum('bij,bij->bj', alpha, x_dense)
            ts.append(t)
            h = self.gru_cell(t, h)
        
        ts = torch.stack(ts, dim=1)
        
        # Process with Mamba
        x_mamba = self.mamba(inputs_embeds=ts).last_hidden_state
        x_mamba = self.layer_norm_mamba(x_mamba)

        # Merge with original node features
        x_m = x_mamba[batch]
        x = self.merge(torch.cat([x_orig, x_m[:, -1, :]], dim=1))
        
        x = self.output_gin(x, edge_index)
        
        # Apply readout if specified
        if self.readout is not None:
            x = self.readout(x, batch)

        if self.dec is not None:
            x = self.dec(x)
        
        return x
