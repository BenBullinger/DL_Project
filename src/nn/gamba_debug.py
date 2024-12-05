import torch
import torch.nn as nn
from torch_geometric.nn import GINConv
from torch_geometric.data import Data, Batch
from transformers import MambaConfig, MambaModel
from torch_scatter import scatter_mean

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

class TokenAggregator(nn.Module):
    """Aggregate graph information into a fixed set of tokens."""
    def __init__(self):
        super().__init__()

    def forward(self, x, batch=None):
        if batch is not None:
            return scatter_mean(x, batch, dim=0)
        else:
            return x.mean(dim=1)

class GraphAttentionAggregator(nn.Module):
    """Aggregates graph information into a fixed number of nodes using transformer attention"""
    def __init__(self, hidden_channels, num_virtual_tokens, num_attention_heads=1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=num_attention_heads,
            batch_first=True
        )

        self.virtual_tokens = torch.randn(num_virtual_tokens, hidden_channels)

        self.aggr = TokenAggregator()
        
    def forward(self, x, virtual_tokens, batch):
        batch_size = torch.max(batch) + 1

        virtual_tokens = self.virtual_tokens.unsqueeze(0).expand(batch_size, -1, -1).clone()
        #virtual_tokens.shape = [batch, num_virtual_tokens, hidden_channel]
        aggregated_tokens = self.aggr(x, batch)
        aggregated_tokens = aggregated_tokens.unsqueeze(1) # [batch_size, channels]
        
        x_batched = torch.zeros(batch_size, max(batch.bincount()), x.size(-1), device=x.device)
        input(f"Virtual tokens at start:\n{virtual_tokens[0,:,:]}")
        for i in range(self.virtual_tokens.size(0)):
            query = aggregated_tokens
            out, _ = self.attention(
                query=query,
                key=x_batched,
                value=x_batched
            )
            aggregated_tokens = self.aggr(torch.cat((virtual_tokens[:,0:i,:], out), dim=1)).unsqueeze(1)
            virtual_tokens[:,i,:] = aggregated_tokens.squeeze(1)
            input(f"Virtual tokens after iteration {i}:\n{virtual_tokens[0,:,:]}")
        pass  # [batch_size, num_virtual_tokens, channels]

class Gamba(nn.Module):
    def __init__(self, hidden_channels, virtual_tokens):
        super().__init__()
        self.gaa = GraphAttentionAggregator(hidden_channels, virtual_tokens)
        self.virtual_tokens = virtual_tokens

    def forward(self, x, edge_index, batch):
        self.gaa(x, self.virtual_tokens, batch)
        

if __name__ == '__main__':
    edge_index1 = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t()
    x1 = torch.tensor([[1,0], [2,0]], dtype=torch.float)

    graph1 = Data(x=x1, edge_index=edge_index1)

    # Graph 2: 3 nodes, 2 edges
    edge_index2 = torch.tensor([[0, 1, 1], [1, 2, 0]], dtype=torch.long)
    x2 = torch.tensor([[3,3], [4,0], [5,0]], dtype=torch.float)

    graph2 = Data(x=x2, edge_index=edge_index2)

    # Batch the graphs
    batch = Batch.from_data_list([graph1, graph2])
    # Print the batch to debug
    model = Gamba(2,4)
    model(batch.x, batch.edge_index, batch.batch)