import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
import utils


class DQNAgent(nn.Module):
    def __init__(self):
        super(DQNAgent, self).__init__()
        
        self.fc = nn.Sequencial(
            nn.Linear(100, 40),
            nn.ReLU(),
            nn.Linear(40, 10),
            nn.ReLU(),
            nn.Linear(10, 4),
            nn.Tanh()
        )
        
        
    def forward(self, state: np.ndarray, device: torch.device):
        '''
        Pass a map through the DQN
        '''
        # Prep map for fc layer
        x = state.flatten()
        x = torch.from_numpy(x)
        x.to(device)
        
        out = self.fc(x)
        return torch.round(out)
    
    
    def __call__(self, map: np.ndarray) -> Tuple[int, int]:
        super(DQNAgent, self).__call__()
        return self.forward(self, device=utils.get_device(), map=map)
        
print()