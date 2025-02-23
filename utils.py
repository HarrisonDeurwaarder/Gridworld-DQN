import torch
import numpy as np
from collections import deque


def get_device():
    '''
    Allows use of CUDA if available, otherwise, defaults to CPU
    '''
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)
        
        
    def __len__(self) -> int:
        return len(self.buffer)
    
    
    def add(self, state: torch.tensor, action: int, reward: int, next_state: torch.tensor) -> None:
        self.buffer.append((state, action, reward, next_state))
        
        
    def sample(self, batch_size: int) -> torch.tensor:
        batch = np.random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return (
            torch.tensor(states),
            torch.tensor(actions),
            torch.tensor(rewards),
            torch.tensor(next_states)
        )
        
print()