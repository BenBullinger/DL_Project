import torch
import torch.nn as nn
from torch_geometric.nn import GatedGraphConv
from mamba_ssm import Mamba
from torch_geometric.utils import to_dense_batch
from .gamba_PVOC_multi import SpatialConv
from .mlp import get_mlp

class MultiScaleARSPBlock(nn.Module):
    def __init__(self, hidden_channels, num_virtual_tokens=4, num_attention_heads=4, args=None):
        super().__init__()
        self.num_virtual_tokens = num_virtual_tokens
        # -- Multi-scale SpatialConvs (like GambaSP)
        spatial_convs = 3
        self.spatial_convs = nn.ModuleList([
            SpatialConv(hidden_channels, args) for _ in range(spatial_convs)  # or however many scales you want
        ])
        self.gated_gcn = GatedGraphConv(out_channels=hidden_channels, num_layers=8)
        
        # -- GRU approach for virtual tokens (like GambaAR)      
        self.gru_cell = nn.GRUCell(hidden_channels*2, hidden_channels)
        weights = get_mlp(
            input_dim=hidden_channels*3, hidden_dim=hidden_channels*2,
            mlp_depth=3, output_dim=1,
            normalization=nn.Identity, last_relu=False
        )
        self.weights = nn.Sequential(weights, nn.Sigmoid()) 
        # If you want a bigger dimension for the GRU, you can adjust above to "hidden_channels * 2", etc.

        # -- Mamba config
        self.mamba = Mamba(
            d_model = hidden_channels*2,
            d_state = 128,
            d_conv = 4,
            expand = 2
        )

        # -- Merge multi-scale features + final Mamba token
        # Suppose we have 8 SpatialConv outputs + 1 from Mamba = 9 × hidden_channels
        self.merge = get_mlp(
            input_dim=hidden_channels * (spatial_convs+2),  
            hidden_dim=hidden_channels,
            output_dim=hidden_channels,
            mlp_depth=1,
            normalization=nn.LayerNorm,
            last_relu=False
        )
        
        self.layer_norm_mamba = nn.LayerNorm(hidden_channels*2)
        self.layer_norm = nn.LayerNorm(hidden_channels)

    def forward(self, x, edge_index, edge_attr, batch, **kwargs):
            """
            x:         [N, hidden_channels]
            edge_index:[2, E]
            edge_attr: [E, num_edge_features]  (e.g. 2 if sobel + boundary)
            batch:     [N]  with batch indices
            """
            #input(f"{x.shape}, {edge_attr.shape}")
            identity = x
            pe = self.gated_gcn(x, edge_index)
            probabilistic = False
            #Spatial conv thing
            xs = []
            for conv in self.spatial_convs:
                xs.append(conv(x, edge_index, edge_attr))

            x = torch.cat([x, pe], dim=1)
            #Mamba
            x_dense, mask = to_dense_batch(x, batch)  
            B, maxN, C = x_dense.shape
            h = torch.zeros(B, int(C/2), device=x.device)  

            tokens = []
            num_tokens = self.num_virtual_tokens if self.training else self.num_virtual_tokens
            for _ in range(num_tokens):
                #input(torch.cat([x_dense, h.unsqueeze(1).expand(-1, x_dense.shape[1], -1)], dim=-1).shape)
                probs = self.weights(torch.cat([x_dense, h.unsqueeze(1).expand(-1, x_dense.shape[1], -1)], dim=-1))
                #input(f"{probs.shape}, {x_dense.shape}")
                alpha = torch.bernoulli(probs) if probabilistic else probs
                #input(f"{alpha.shape}, {x_dense.shape}")
                t = (alpha * x_dense).mean(dim=1)
                #input(f"{t.shape}, {h.shape}")
                h = self.gru_cell(t, h)  
                tokens.append(t) 

            tokens = torch.stack(tokens, dim=1)
            #input(tokens.shape)
            mamba_output = self.mamba(tokens)  
            mamba_output = self.layer_norm_mamba(mamba_output)  

            x_m = mamba_output[:, -1, :]  

            x_m_node = x_m[batch]

            cat_all = torch.cat([*xs, x_m_node], dim=-1)  
            x_new = self.merge(cat_all)  

            out = self.layer_norm(identity + x_new)

            return out

import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from .mlp import get_mlp

class GambaARSP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        layers=3,
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
        
        print("GambaARSP with", layers, "layers")
        self.num_layers = layers
        self.args = args

        # 1) Optional encoder
        self.enc = None
        if use_enc:
            self.enc = get_mlp(
                input_dim=in_channels,
                hidden_dim=hidden_channels,
                mlp_depth=mlp_depth,
                output_dim=hidden_channels,
                normalization=nn.LayerNorm,
                last_relu=False
            )
        
        # 2) Stacked MultiScaleARSPBlock layers
        self.layers = nn.ModuleList([
            MultiScaleARSPBlock(
                hidden_channels=hidden_channels,
                num_virtual_tokens=num_virtual_tokens,
                num_attention_heads=num_attention_heads,
                args=args
            )
            for _ in range(layers)
        ])

        # 3) Optional decoder
        self.dec = None
        if use_dec:
            self.dec = get_mlp(
                input_dim=hidden_channels,
                hidden_dim=hidden_channels,
                mlp_depth=mlp_depth,
                output_dim=out_channels,
                normalization=nn.LayerNorm,
                last_relu=False
            )
        
        # 4) Readout
        self.readout = None
        if use_readout:
            supported_pools = {
                'add': global_add_pool,
                'mean': global_mean_pool,
                'max': global_max_pool
            }
            self.readout = supported_pools.get(use_readout, global_add_pool)

    def forward(self, x, edge_index, batch, edge_attr=None, **kwargs):
        # -- (Optional) Encode
        if self.enc is not None:
            x = self.enc(x)

        # -- Apply each ARSP block
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, batch)

        # -- Readout
        if self.readout is not None:
            x = self.readout(x, batch)
        
        # -- (Optional) Decode
        if self.dec is not None:
            x = self.dec(x)

        return x