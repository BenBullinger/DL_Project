import torch
import torch.nn as nn
#from transformers import MambaConfig, MambaModel
from mamba_ssm import Mamba
from torch_geometric.utils import to_dense_batch
from .gamba_PVOC_multi import SpatialConv
from .mlp import get_mlp

class MultiScaleARSPBlock(nn.Module):
    def __init__(self, hidden_channels, num_virtual_tokens=4, num_attention_heads=4):
        super().__init__()
        self.num_virtual_tokens = num_virtual_tokens
        # -- Multi-scale SpatialConvs (like GambaSP)
        self.spatial_convs = nn.ModuleList([
            SpatialConv(hidden_channels) for _ in range(8)  # or however many scales you want
        ])
        
        # -- GRU approach for virtual tokens (like GambaAR)
        # We will produce embeddings from the (x, multi-scale features) and feed them to a GRU in a loop.
        # "x" will have shape [N, hidden_channels]; we might project it to [N, something] then do a to_dense_batch...
        
        self.gru_cell = nn.GRUCell(hidden_channels, hidden_channels)  
        # If you want a bigger dimension for the GRU, you can adjust above to "hidden_channels * 2", etc.

        # -- Mamba config
        self.mamba = Mamba(
            d_model = hidden_channels,
            d_state = 128,
            d_conv = 4,
            expand = 2
        )

        # -- Merge multi-scale features + final Mamba token
        # Suppose we have 8 SpatialConv outputs + 1 from Mamba = 9 × hidden_channels
        self.merge = get_mlp(
            input_dim=hidden_channels * 9,  
            hidden_dim=hidden_channels,
            output_dim=hidden_channels,
            mlp_depth=1,
            normalization=nn.LayerNorm,
            last_relu=False
        )
        
        self.layer_norm = nn.LayerNorm(hidden_channels)

    def forward(self, x, edge_index, edge_attr, batch):
        """
        x:         [N, hidden_channels]
        edge_index:[2, E]
        edge_attr: [E, num_edge_features]  (e.g. 2 if sobel + boundary)
        batch:     [N]  with batch indices
        """
        identity = x
        
        # 1) Multi-scale spatial processing
        xs = []
        for conv in self.spatial_convs:
            xs.append(conv(x, edge_index, edge_attr))
        
        # 2) GRU-based virtual token approach
        #    We'll transform x -> dense batch -> iterative GRU -> Mamba
        x_dense, mask = to_dense_batch(x, batch)  # [B, maxN, hidden_channels]
        B, maxN, C = x_dense.shape
        
        # We'll keep a hidden state h per batch example. shape: [B, hidden_channels]
        h = torch.zeros(B, C, device=x.device)  # initial zero hidden states

        # Collect the tokens from each iteration
        tokens = []
        num_tokens = self.num_virtual_tokens if self.training else 8*self.num_virtual_tokens
        for _ in range(num_tokens):
            # A simple approach: do a mean or sum over the node features per graph, and combine with h
            # Or you can do something more advanced. For simplicity, let's do a mean:
            x_mean = x_dense.mean(dim=1)  # [B, hidden_channels]
            
            # Combine x_mean with the previous hidden state h in some manner
            # For example, GRUCell input is [B, input_size], so we can just pass x_mean
            h = self.gru_cell(x_mean, h)  # [B, hidden_channels]
            tokens.append(h.unsqueeze(1)) # store [B, 1, hidden_channels]
        
        # Stack all tokens => [B, num_virtual_tokens, hidden_channels]
        tokens = torch.cat(tokens, dim=1)

        # 3) Mamba
        #    Mamba expects something like [B, seq_len, hidden_size]
        #    Here: [B, num_virtual_tokens, hidden_channels]
        mamba_output = self.mamba(tokens)  # [B, num_virtual_tokens, hidden_channels]
        mamba_output = self.layer_norm_mamba(mamba_output)  # optional LN

        # We'll take the last token as a summary vector for each batch
        # or you can do any pooling across tokens
        x_m = mamba_output[:, -1, :]  # [B, hidden_channels]

        # Expand x_m back to node level, mapping each node's "batch b" -> x_m[b]
        # shape => [N, hidden_channels]
        x_m_node = x_m[batch]

        # 4) Merge all features
        #    Concatenate all 8 SpatialConv outputs plus x_m_node => 9 × hidden_channels
        cat_all = torch.cat([*xs, x_m_node], dim=-1)  # [N, hidden_channels*9]
        x_new = self.merge(cat_all)  # [N, hidden_channels]
        
        # 5) Residual + LN
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
                num_attention_heads=num_attention_heads
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