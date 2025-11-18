import torch
import torch.nn as nn

class Backbone(nn.Module):
    def __init__(self, input_size: int, hidden_dim: tuple[int, int]):
        super().__init__()
        depth, width = hidden_dim
        
        layers = [
            nn.Linear(input_size, width), 
            nn.ReLU(inplace=True)
        ]
        
        for _ in range(depth):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU(inplace=True))
            
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)