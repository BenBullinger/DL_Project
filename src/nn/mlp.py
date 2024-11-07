import torch 
from torch import nn
from torch import Tensor
from torch_geometric.nn.resolver import activation_resolver
import torch.nn.functional as F
from typing import List, Union, Optional, Dict, Any

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

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Union[int, List[int]],
        num_hidden_layers: int = 1,
        dropout: float = 0.0,
        act: str = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Multi-Layer Perceptron (MLP) module.

        Args:
            input_dim (int): Dimensionality of the input features.
            output_dim (int): Dimensionality of the output features.
            hidden_dims (Union[int, List[int]]): Hidden layer dimensions.
                If int, same hidden dimension is used for all layers.
            num_hidden_layers (int, optional): Number of hidden layers. Default is 1.
            dropout (float, optional): Dropout probability. Default is 0.0.
            act (str, optional): Activation function name. Default is "relu".
            act_kwargs (Dict[str, Any], optional): Additional arguments for the activation function.
                                                   Default is None.
        """
        super(MLP, self).__init__()

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims] * num_hidden_layers

        assert len(hidden_dims) == num_hidden_layers

        hidden_dims = [input_dim] + hidden_dims
        layers = []

        for i_dim, o_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers.append(nn.Linear(i_dim, o_dim, bias=True))
            layers.append(activation_resolver(act, **(act_kwargs or {})))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))

        layers.append(nn.Linear(hidden_dims[-1], output_dim, bias=True))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the MLP module.

        Args:
            x (Any): Input tensor.

        Returns:
            Any: Output tensor.
        """
        return self.mlp(x)