import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
import utils


class DQNAgent(nn.Module):
    def __init__(self) -> None:
        super(DQNAgent, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(100, 40),
            nn.ReLU(),
            nn.Linear(40, 10),
            nn.ReLU(),
            nn.Linear(10, 4),
            nn.Tanh()
        )
        
        
    def forward(self, state: np.ndarray, device: torch.device = torch.device('cpu')) -> torch.Tensor:
        '''
        Pass a state through the DQN
        '''
        # Prep map for fc layer
        x = torch.from_numpy(state)
        x = x.float()
        x = x.flatten()
        x.to(device)
        
        return self.fc(x)
    
    
    def __call__(self, state: np.ndarray) -> int:
        q_vals = super(DQNAgent, self).__call__(state, utils.get_device())
        
        return torch.argmax(q_vals).item()