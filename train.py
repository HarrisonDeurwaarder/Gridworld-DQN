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
EPSILON_DECAY_RATE, epsilon, EPSILON_MIN = 0.005, 1.0, 0.05
    
    
def main() -> None:
    # Initialize prey and target prey DQNs
    policy_prey, target_prey = model.DQNAgent(), model.DQNAgent()
    
    # Initialize predator and target predator DQNs
    policy_pred, target_pred = model.DQNAgent(), model.DQNAgent()
    
    buffer = utils.ReplayBuffer(BUFFER_SIZE)
    prey_optimizer, pred_optimizer = optim.Adam(), optim.Adam()
    criterion = nn.MSELoss()
    
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
            target_prey = model.DQNAgent.load_state_dict(policy_prey.state_dict())
            target_pred = model.DQNAgent.load_state_dict(policy_pred.state_dict())
        
        state, next_state, prey_r, pred_r = episode.step()
        buffer.add()
        
        # Skip training if insufficient number of steps are present
        if len(buffer) < BATCH_SIZE:
            continue
        
        states, actions, rewards
        
        # Backpropagate through the prey DQN
        prey_q_vals = target_prey(state)
        target = prey_reward + DISCOUNT_FACTOR * 1
        
        prey_loss = criterion(max(prey_q_vals), target)

        
        step_count += 1
        epsilon -= EPSILON_DECAY_RATE if epsilon > EPSILON_MIN else 0
        

if __name__ == '__main__':
    main()