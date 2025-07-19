
import torch
import torch.nn as nn

class EMWaveNet(nn.Module):
    """
    Neural network for electromagnetic wave fields
    Outputs: [Ex_real, Ex_imag, Ey_real, Ey_imag, Ez_real, Ez_imag]
    """
    def __init__(self, layers=[3, 64, 64, 64, 64, 6], activation_type='tanh'):
        super(EMWaveNet, self).__init__()
        
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        
        # Choose activation function
        if activation_type == 'tanh':
            self.activation = nn.Tanh()
        elif activation_type == 'sine':
            self.activation = lambda x: torch.sin(30 * x)  # Sine activation for periodic solutions
        elif activation_type == 'swish':
            self.activation = lambda x: x * torch.sigmoid(x)  # Swish activation
        else:
            self.activation = nn.Tanh()
            
        self.init_weights()
    
    def init_weights(self):
        """Xavier initialization"""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, x):
        """Forward pass: x = [x, y, z]"""
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
        x = self.layers[-1](x)  # No activation on output
        return x
