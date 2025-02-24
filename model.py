import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
import utils


class DQNAgent(nn.Module):
    def __init__(self) -> None:
        super(DQNAgent, self).__init__()
        
        self.fc = nn.Sequencial(
            nn.Linear(100, 40),
            nn.ReLU(),
            nn.Linear(40, 10),
            nn.ReLU(),
            nn.Linear(10, 4),
            nn.Tanh()
        )
        
        
    def forward(self, state: np.ndarray, device: torch.device, epsilon: float = 0.05) -> torch.Tensor:
        '''
        Pass a map through the DQN
        '''
        # Explore
        if np.random.random() <= epsilon:
            return np.random.randint(0, 4)
        
        # Exploit
        else:
            # Prep map for fc layer
            x = state.flatten()
            x = torch.from_numpy(x)
            x.to(device)
            
            return self.fc(x)
    
    
    def __call__(self, state: np.ndarray) -> int:
        super(DQNAgent, self).__call__()
        
        q_vals = self.forward(self, device=utils.get_device(), state=state)
        return torch.argmax(q_vals).item()