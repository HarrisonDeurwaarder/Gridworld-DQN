import model, environment, utils
import torch
import torch.nn as nn
import torch.optim as optim
    
    
STEPS = 500
BUFFER_SIZE = 100
TARGET_NET_UPDATE_FREQ = 100
LR = 1e-3
BATCH_SIZE = 8
DISCOUNT_FACTOR = 0.99
EPSILON_DECAY_RATE, EPSILON_MIN = 0.005, 0.05
    
    
def main() -> None:
    # Initialize prey and target prey DQNs
    prey = model.DQNAgent()
    target_prey = model.DQNAgent.load_state_dict(prey.state_dict())
    
    # Initialize predator and target predator DQNs
    pred = model.DQNAgent()
    target_pred = model.DQNAgent.load_state_dict(pred.state_dict())
    
    buffer = utils.ReplayBuffer(BUFFER_SIZE)
    optimizer = optim.Adam()
    
    episode = environment.Env(*environment.Env.gen_grid(), 
                              terminal_reward=10, 
                              distance_scale_factor=0.5)
    step_count = 0
    while step_count < STEPS:
        # Check if the current episode is done, then reset episode
        if episode.done:
            episode = environment.Env(*environment.Env.gen_grid(), 
                                    terminal_reward=10, 
                                    distance_scale_factor=0.5)
        # Update the target networks every x steps
        if step_count % TARGET_NET_UPDATE_FREQ == 0:
            target_prey = model.DQNAgent.load_state_dict(prey.state_dict())
            target_pred = model.DQNAgent.load_state_dict(prey.state_dict())
        
        # Skip training if insufficient number of steps are present
        if len(buffer) < BATCH_SIZE:
            continue

        
        step_count += 1
        

if __name__ == '__main__':
    main()