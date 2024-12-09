# Gamba without Mamba => Ga

from src.nn.gamba import Gamba

class Ga(Gamba):
    def forward(self, x, edge_index, batch, **kwargs):
        x = self.input_gin(x, edge_index)
        
        for layer in self.layers:
            # Regular message passing
            gin_out = layer['gin'](x, edge_index)
            
            # Global information processing
            global_tokens = layer['attention'](x, batch)
            global_update = global_tokens[:, -1, :]  # Use only the last global token
            
            # Normalized feature combination
            x = self.layer_norm(gin_out + 0.1 * global_update[batch])  # Scale global contribution
            
        x = self.output_gin(x, edge_index)
        
        # Apply readout if specified
        if self.readout is not None:
            x = self.readout(x, batch)
        
        return x