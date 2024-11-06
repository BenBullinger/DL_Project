import copy

import torch 
import torch.nn.functional as F

from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, global_max_pool


supported_norms = {'layernorm': torch.nn.LayerNorm, 'batchnorm':torch.nn.BatchNorm1d, 'none':torch.nn.Identity}
supported_pools = {'add': global_add_pool, 'mean': global_mean_pool, 'max': global_max_pool}

def get_mlp(input_dim, hidden_dim, mlp_depth, output_dim, normalization, dropout=0.0, last_relu = True):
    """
    Creates a classical MLP. The first layer is input_dim x hidden_dim, the last layer is hidden_dim x output_dim. The rest is hidden x hidden
    For normalization, pass the corresponding torch.nn function from here: https://pytorch.org/docs/stable/nn.html#normalization-layers
    (e.g.: get_mlp(5,10,4,3, torch.nn.LayerNorm))
    """
    relu_layer = torch.nn.ReLU()
    modules = [torch.nn.Linear(input_dim, int(hidden_dim)), normalization(int(hidden_dim)), relu_layer, torch.nn.Dropout(dropout)]
    for i in range(0, int(mlp_depth)):
        modules = modules + [torch.nn.Linear(int(hidden_dim), int(hidden_dim)), normalization(int(hidden_dim)), relu_layer, torch.nn.Dropout(dropout)]
    modules = modules + [torch.nn.Linear(int(hidden_dim), output_dim)]
    
    if last_relu:
        modules.append(normalization(output_dim))
        modules.append(relu_layer)

    return torch.nn.Sequential(*modules)
  

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, layers, out_channels, mlp_depth, normalization, dropout, use_enc, use_dec, use_readout=None):
        super(GIN, self).__init__()
        norm_func = None
        if normalization in supported_norms.keys():
            norm_func = supported_norms[normalization]
        else:
            print(f"Your normalization argument \"{normalization}\" is either not valid or not supported.")
            exit(1)
        
        if use_enc:
            self.enc = get_mlp(input_dim=in_channels, hidden_dim=hidden_channels, mlp_depth=mlp_depth, output_dim=hidden_channels, normalization=norm_func, last_relu=False)
            self.conv = torch.nn.ModuleList([GINConv(get_mlp(input_dim=hidden_channels, hidden_dim=hidden_channels, mlp_depth=mlp_depth, output_dim=hidden_channels, normalization=norm_func, dropout=dropout)) for _ in range(layers-1)])
            if use_dec:
                self.conv.append(GINConv(get_mlp(input_dim=hidden_channels, hidden_dim=hidden_channels, mlp_depth=mlp_depth, output_dim=hidden_channels, normalization=norm_func, dropout=dropout)))   
                self.dec = get_mlp(input_dim=hidden_channels, hidden_dim=hidden_channels, mlp_depth=mlp_depth, output_dim=out_channels, normalization=norm_func, last_relu=False)
            else:
                self.conv.append(GINConv(get_mlp(input_dim=hidden_channels, hidden_dim=hidden_channels, mlp_depth=mlp_depth, output_dim=out_channels, normalization=norm_func, dropout=dropout)))
        
        self.use_readout = use_readout is not None
        if self.use_readout:
            self.readout = supported_pools.get(use_readout, global_add_pool)


    def forward(self, x, edge_index, batch):
        if self.enc is not None:
            x = self.enc(x)
        
        for conv in self.conv:
            x = conv(x, edge_index)

        if self.readout is not None:
            x = self.readout(x, batch)

        if self.dec is not None:
            x = self.dec(x)
        
        return x
        