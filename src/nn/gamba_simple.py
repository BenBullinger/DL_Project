import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, GatedGraphConv
#from transformers import MambaConfig, MambaModel
from torch_geometric.utils import to_dense_batch
from mamba_ssm import Mamba
from .mlp import get_mlp
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

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
        
        self.input_gin = GINConv(get_mlp(input_dim=first_gin_dim, hidden_dim=hidden_channels, mlp_depth=mlp_depth, output_dim=hidden_channels, normalization=torch.nn.LayerNorm, last_relu=False), train_eps=True)
        
        self.pe_dims = args.laplacePE + args.RW_length
        self.pe_gnn = GatedGraphConv(out_channels=hidden_channels, num_layers=8)

        self.theta = nn.Linear(hidden_channels+self.pe_dims, num_virtual_tokens, bias=False)
        # Configure Mamba
        self.mamba = Mamba(
            d_model = hidden_channels+self.pe_dims,
            d_state = 128,
            d_conv = 4,
            expand = 2
        )
        self.layer_norm_mamba = nn.LayerNorm(hidden_channels+self.pe_dims)
        
        self.merge = get_mlp(input_dim=hidden_channels*2+self.pe_dims, hidden_dim=hidden_channels, output_dim=hidden_channels, mlp_depth=1, normalization=torch.nn.LayerNorm, last_relu=False)
            
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
        
        
    def forward(self, x, edge_index, batch, **kwargs):
        lpe, rwse = kwargs["laplacePE"], kwargs["rwse"]
        if self.enc is not None:
            x = self.enc(x)
        x_orig =x
        pe = torch.cat([lpe, rwse], dim=1)
        #pe = self.pe_gnn(x, edge_index)
        #x = self.input_gin(x, edge_index)

        x = torch.cat([x, pe], dim=1)
        x_dense, mask = to_dense_batch(x, batch)
        alpha = torch.sigmoid(self.theta(x_dense))
        #alpha = torch.softmax(self.theta(x_dense), dim=1)
        #alpha_X = alpha.transpose(1,2) @ x_dense
        #x_mamba = self.mamba(alpha_X)
        
        mask = torch.bernoulli(alpha)
        #mask = top_k_mask(alpha=alpha, k=1)
        #mask = (alpha > 0.7).float()
        tokens = (mask.unsqueeze(-1) * x_dense.unsqueeze(2)).mean(dim=1)
        x_mamba = self.mamba(tokens)
        x_mamba = self.layer_norm_mamba(x_mamba)

        x_m = x_mamba[batch]
        x = self.merge(torch.cat([x_orig, x_m[:,-1,:]], dim=1)) 
        
        x = self.output_gin(x, edge_index)
        
        # Apply readout if specified
        if self.readout is not None:
            x = self.readout(x, batch)

        if self.dec is not None:
            x = self.dec(x)
        
        return x
    
def top_k_mask(alpha: torch.Tensor, k: int) -> torch.Tensor:
    """
    alpha: Tensor of shape [B, N, K]
      B = batch size
      N = number of nodes
      K = number of "slots" or channels
    k: number of top elements to keep in each slot

    Returns a binary mask of the same shape [B, N, K],
    where for each (batch b, slot i), exactly k entries
    along dim=1 (the node dimension) are set to 1, and
    all others are 0.
    """
    B, N, K = alpha.shape
    mask = torch.zeros_like(alpha)

    # For each slot i, pick top k along dim=1 (the node dimension)
    for i in range(K):
        _, idx = alpha[:, :, i].topk(k, dim=1)  # idx shape: [B, k]
        # We index into 'mask[b, idx[b, j], i]' and set those to 1
        b_idx = torch.arange(B, device=alpha.device).unsqueeze(1).expand(B, k)
        mask[b_idx, idx, i] = 1.0

    return mask