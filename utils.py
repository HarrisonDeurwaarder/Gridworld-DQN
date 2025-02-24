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
    
    
    def add(self, state: torch.tensor, next_state: torch.tensor, prey_action: int, pred_action: int, prey_reward: float, pred_reward: float) -> None:
        self.buffer.append((state, next_state, prey_action, pred_action, prey_reward, pred_reward))
        
        
    def sample(self, batch_size: int) -> torch.tensor:
        batch = np.random.sample(self.buffer, batch_size)
        states, next_states, prey_actions, pred_actions, prey_rewards, pred_rewards = zip(*batch)
        return (
            torch.tensor(states),
            torch.tensor(next_states),
            torch.tensor(prey_actions),
            torch.tensor(pred_actions),
            torch.tensor(prey_rewards),
            torch.tensor(pred_rewards),
        )