import torch
import numpy as np
import random as r
from collections import deque


def get_device():
    '''
    Allows use of CUDA if available, otherwise, defaults to CPU
    '''
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def verify_shift(state: np.ndarray, point: tuple, shift: tuple, value: tuple) -> np.ndarray:
    '''
    Attempts to perform a shift of a point through a state, unless it encounters an edge
    '''
    x_candidate, y_candidate = point[0] + shift[0], point[1] + shift[1]
    return (x_candidate if x_candidate < state.shape[0] and x_candidate > 0 else point[0], 
            y_candidate if y_candidate < state.shape[1] and y_candidate > 0 else point[1])
    
    

class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)
        
        
    def __len__(self) -> int:
        return len(self.buffer)
    
    
    def add(self, state: torch.tensor, next_state: torch.tensor, prey_action: int, pred_action: int, prey_reward: float, pred_reward: float) -> None:
        self.buffer.append((state, next_state, prey_action, pred_action, prey_reward, pred_reward))
        
        
    def sample(self, batch_size: int) -> torch.tensor:
        batch = r.sample(self.buffer, batch_size)
        states, next_states, prey_actions, pred_actions, prey_rewards, pred_rewards = zip(*batch)
        return (
            np.array(states),
            np.array(next_states),
            np.array(prey_actions),
            np.array(pred_actions),
            np.array(prey_rewards),
            np.array(pred_rewards),
        )