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
        self.args = args
        self.num_virtual_tokens = num_virtual_tokens
        # -- Multi-scale SpatialConvs (like GambaSP)
        self.d = hidden_channels
        spatial_convs = 3
        self.spatial_convs = nn.ModuleList([
            SpatialConv(hidden_channels, args) for _ in range(spatial_convs)  # or however many scales you want
        ])
        pe_dim = hidden_channels if args.pe == "gnn" else args.laplacePE + args.RW_length
        if args.pe == "gnn":
            self.gated_gcn = GatedGraphConv(out_channels=hidden_channels, num_layers=8)
        
        # -- GRU approach for virtual tokens (like GambaAR)      
        self.gru_cell = nn.GRUCell(hidden_channels + pe_dim, hidden_channels + pe_dim)
        weights = get_mlp(
            input_dim=(hidden_channels+ pe_dim)*2 , hidden_dim=hidden_channels + pe_dim,
            mlp_depth=3, output_dim=1,
            normalization=nn.Identity, last_relu=False
        )
        self.weights = nn.Sequential(weights)#, nn.Sigmoid()) 
        # If you want a bigger dimension for the GRU, you can adjust above to "hidden_channels * 2", etc.

        # -- Mamba config
        self.mamba = Mamba(
            d_model = hidden_channels + pe_dim,
            d_state = 128,
            d_conv = 4,
            expand = 2
        )

        # -- Merge multi-scale features + final Mamba token
        # Suppose we have 8 SpatialConv outputs + 1 from Mamba = 9 × hidden_channels
        self.merge = get_mlp(
            input_dim=hidden_channels * (spatial_convs+1) + pe_dim,  
            hidden_dim=hidden_channels,
            output_dim=hidden_channels,
            mlp_depth=1,
            normalization=nn.LayerNorm,
            last_relu=False
        )
        
        self.layer_norm_mamba = nn.LayerNorm(hidden_channels+ pe_dim)
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
            lpe, rwse = kwargs["laplacePE"], kwargs["rwse"]
            x_orig =x
            #pe = torch.cat([lpe, rwse], dim=1)
            #Spatial conv thing
            xs = []
            for conv in self.spatial_convs:
                xs.append(conv(x, edge_index, edge_attr))
            
            if self.args.pe == "gnn":
                pe = self.pe_gnn(x, edge_index)
                x = torch.cat([x, pe], dim=1)
            else:
                x = torch.cat([x, lpe, rwse], dim=1)
            #Mamba
            x_dense, mask = to_dense_batch(x, batch)  
            B, maxN, C = x_dense.shape
            h = torch.zeros(B, x.shape[1], device=x.device)  

            tokens = []
            num_tokens = self.num_virtual_tokens if self.training else self.num_virtual_tokens
            for _ in range(num_tokens):
                alpha = self.weights(torch.cat([x_dense, h.unsqueeze(1).expand(-1, x_dense.shape[1], -1)], dim=-1))
                if self.args.regularization == "attention":
                    inv_sqrt_d = self.d ** -0.5
                    alpha = torch.softmax(inv_sqrt_d * alpha, dim=1)
                elif self.args.regularization == "probabilistic":
                    alpha = torch.bernoulli(torch.sigmoid(alpha))
                t = (alpha * x_dense).mean(dim=1)
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
            x = layer(x, edge_index, edge_attr, batch, **kwargs)

        # -- Readout
        if self.readout is not None:
            x = self.readout(x, batch)
        
        # -- (Optional) Decode
        if self.dec is not None:
            x = self.dec(x)

        return x