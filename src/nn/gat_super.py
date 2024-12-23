import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_add_pool, global_mean_pool, global_max_pool

from .mlp import get_mlp

supported_norms = {'layernorm': torch.nn.LayerNorm, 'batchnorm': torch.nn.BatchNorm1d, 'none': torch.nn.Identity}
supported_pools = {'add': global_add_pool, 'mean': global_mean_pool, 'max': global_max_pool}

class GATSuper(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, layers, out_channels, heads, normalization, dropout, use_enc, use_dec, use_readout=None):
        super(GATSuper, self).__init__()
        norm_func = None
        if normalization in supported_norms.keys():
            norm_func = supported_norms[normalization]
        else:
            print(f"Your normalization argument \"{normalization}\" is either not valid or not supported.")
            exit(1)

        if use_enc:
            self.enc = get_mlp(input_dim=in_channels, hidden_dim=hidden_channels, mlp_depth=1, output_dim=hidden_channels, normalization=norm_func, last_relu=False)

            self.conv = torch.nn.ModuleList([
                GATConv(
                    in_channels=hidden_channels * heads if i > 0 else hidden_channels,
                    out_channels=hidden_channels,
                    heads=heads,
                    dropout=dropout
                ) for i in range(layers - 1)
            ])

            if use_dec:
                self.conv.append(GATConv(
                    in_channels=hidden_channels * heads if layers > 0 else hidden_channels,
                    out_channels=hidden_channels,
                    heads=heads,
                    dropout=dropout
                ))
                self.dec = get_mlp(input_dim=hidden_channels * heads, hidden_dim=hidden_channels, mlp_depth=1, output_dim=out_channels, normalization=norm_func, last_relu=False)
            else:
                self.conv.append(GATConv(
                    in_channels=hidden_channels * heads,
                    out_channels=out_channels,
                    heads=1,  # Final output should match out_channels
                    dropout=dropout
                ))

        self.use_readout = use_readout is not None
        if self.use_readout:
            self.readout = supported_pools.get(use_readout, global_add_pool)

    def forward(self, x, edge_index, batch, **kwargs):
        if self.enc is not None:
            x = self.enc(x)

        for conv in self.conv:
            x = conv(x, edge_index)
            x = F.elu(x)  # Apply non-linearity after each GATConv

        if self.use_readout:
            x = self.readout(x, batch)

        if hasattr(self, 'dec') and self.dec is not None:
            x = self.dec(x)

        return x
