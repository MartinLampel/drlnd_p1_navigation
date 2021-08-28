import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_units=[64,32]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        self.fc = nn.ModuleList([nn.Linear(hidden_units[i], hidden_units[i+1]) for i in range(len(hidden_units)-1)])
       
    def forward(self, state):
        """Build a network that maps state -> action values."""
        
        x = state
                             
        for layer in self.fc[:-1]:
            x = layer(x)
            x = F.relu(x)
                             
        x = self.fc[-1](x)
                             
        return x