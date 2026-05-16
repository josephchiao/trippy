import slider
import neural_network as nn
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp
import redone
import random

class RL_trainer:

    def __init__(self, model):
        
        self.model = model
        self.log_std = -3
        self.log_floor = -4
        self.log_ceiling = 2
        self.d_log_std = 0
        self.NN = nn.NeuralNetwork((4, 64, 64, 2), [nn.ReLU, nn.ReLU, [nn.linear, nn.sigmoid]], 'nn_library')
        # self.NN.theta_generate()
        self.X = self.NN.theta_recover()

    def reward(self, state):
        location_cf = 4
        angle_cf = 15
        # - abs(self.model.motor_force)/10
        return  (25 - angle_cf * math.cos(state[1]) - location_cf * (state[0])**2) / 10

    def backward_std(self, action, mu, sigma, advantage):
        epsilon = 1e-8
        action_discrepency = action / 200 + 0.5 - mu

        d_mu = -advantage * (action_discrepency / (sigma ** 2 + epsilon))
        
        # 1. Calculate the gradient for this specific frame
        step_d_log_std = -advantage * ((action_discrepency**2 / (sigma ** 2 + epsilon)) - 1.0)
        
        # 2. THE FIX: Clip it before it rubber-bands!
        if abs(step_d_log_std) > 1.0: 
            step_d_log_std = np.sign(step_d_log_std) * 1.0
            
        # 3. Now safely accumulate it
        self.d_log_std += step_d_log_std
        
        return d_mu   
    
    def train(self, variance = 0):

        gamma = 0.99  # Discount factor (how much we care about the future)
        reward_history = []
        log_std_history = []
        fail_count = 0
        best_reward = 3000
        previously_saved = True
        # rolling_counter = np.zeros(50) # Maybe we need???

        for episode in range(5000):

            # learning_rate = 0.001 * (1.0 - (episode / 10000)) + 0.0001
            learning_rate = 0.00002
            # Reset the simulation to the starting position
            scale_factors = np.array([10.0, 2 * np.pi, 50.0, 50.0])
            
            # Start at a 3-degree wobble, and slowly increase it to an 18-degree wobble over 3000 episodes
            max_wobble = 0.5
            # max_wobble = 0.05
            
            random_angle = np.pi - np.random.choice((-max_wobble, max_wobble))
            random_location = np.random.choice((-0.4, 0.4))


            total_episode_reward = 0

            for side in (-1,1):
                self.model.state = [random_location, side * random_angle, 0, 0]
                t = 0
                done = False
                self.d_log_std = 0
                states_memory = []
                targets_memory = []
                while not done:
                    t += 1
    
                    # nn[0] = V (Score), nn[1] = mu (Action)
                    # Normalize before asking for an action
                    normalized_state = self.model.state / scale_factors

                    NN_output = self.NN.feedforward(normalized_state)[-1][0]
                    self.model.motor_force = (NN_output[1] - 0.5) * 200 + np.exp(self.log_std) * np.random.randn()
                    
                    critic = NN_output[0]
                            
                    # --- THE PHYSICS ENGINE ---
                    # The cart moves for 0.02 seconds using the chosen force
                    next_state = self.model.rk4_step()
                    reward = self.reward(next_state)
                    total_episode_reward += reward
                    # has_nan = np.isnan(next_state).any() 
                    done = next_state[1] <= np.pi/2 or next_state[1] >= 3*np.pi/2 or t >= (self.model.t_end - self.model.t_start) * self.model.fps or abs(next_state[2]) > 100 or abs(next_state[3]) > 100 or abs(next_state[0]) > 6.5                

                    # --- THE TARGET CALCULATION ---
                    # Value of the state we just landed in
                    if done:
                        target_value = reward # If we died, there is no future.
                        if t < (self.model.t_end - self.model.t_start) * self.model.fps:
                            target_value = -50.0   # Punish death before time ends
                        if t < 30:
                            target_value = -100.0  # Punish early death heavily

                    else:
                        normalized_next_state = next_state / scale_factors
                        next_critic = self.NN.feedforward(normalized_next_state)[-1][0][0]
                        target_value = reward + gamma * next_critic
                        
                    # Advantage: Was the move better than the Critic expected?
                    advantage = target_value - critic
                    advantage = np.clip(advantage, -15.0, 15.0)

                    # --- THE BACKWARD PASS ---
                    # 1. Backprop for the Critic (Mean Squared Error)
                    # Loss = 0.5 * (target_value - critic)^2
                    # dL/dV = -(target_value - critic) = -advantage
                    d_V = -advantage
                    
                    # 2. Backprop for the Actor
                    # This function updates self.d_log_std internally and returns the gradient for mu
                    d_mu = self.backward_std(
                        action=self.model.motor_force, 
                        mu=NN_output[1], 
                        sigma=np.exp(self.log_std)/200, 
                        advantage=advantage)

                    # Capping function
                    if abs(d_V) > 1.0: d_V = np.sign(d_V) * 1.0
                    if abs(d_mu) > 0.25: d_mu = np.sign(d_mu) * 0.25              
                    
                    # Move to the next frame
                    self.model.state = next_state

                    states_memory.append(normalized_state)
                    target_V = critic - d_V
                    target_mu = NN_output[1] - d_mu
                    target_mu = np.clip(target_mu, 0.05, 0.95) 
                    
                    targets_memory.append([target_V, target_mu])

                    batch_size = 128
                    
                    if len(states_memory) >= batch_size:
                        self.NN.backward(np.array(states_memory), np.array(targets_memory), learning_rate / batch_size)
                        
                        # 2. Update exploration noise
                        self.log_std -= learning_rate * self.d_log_std / batch_size
                        self.log_std = np.clip(self.log_std, self.log_floor, self.log_ceiling)
                        
                        # 3. Clear the buffers for the next 128 frames
                        states_memory = []
                        targets_memory = []
                        self.d_log_std = 0  

                if len(states_memory) > 0:
                    self.NN.backward(np.array(states_memory), np.array(targets_memory), learning_rate / len(states_memory))
                    self.log_std -= learning_rate * self.d_log_std / len(states_memory)
                    self.log_std = np.clip(self.log_std, self.log_floor, self.log_ceiling)
                if side == -1:
                    t_1 = t

            print(f"Episode {episode} finished! Total Reward: {total_episode_reward:.2f}, runtime = {t_1}, {t}")
            # if t == (self.model.t_end - self.model.t_start) * self.model.fps and t_1 == t:
            #     print('full runtime')
            #     self.NN.theta_save()
            #     previously_saved = True
            #     print('Saved!')

            if total_episode_reward >= best_reward:
                best_reward = total_episode_reward
                self.NN.theta_backup()
                self.NN.theta_save()
                previously_saved = True
                print('Saved!')


            if total_episode_reward < max(70, best_reward * 0.6):
                fail_count += 1
            else:
                fail_count = 0

            if fail_count >= 100 and previously_saved:
                self.NN.theta_recover(i = 1) 
                print('policy_collapse')
                self.log_std = -3
                fail_count = 0           

            reward_history.append(total_episode_reward)
            log_std_history.append(self.log_std)

        # if total_episode_reward >= best_reward:
        #     self.NN.theta_save()
        #     print('Saved!')

        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(reward_history)
        ax2.plot(log_std_history)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Log Std')
        plt.show()

y0 = [0, np.pi-1, 0, 0]
SP = redone.SinglePendulum(params=(9.8, 1, 1, 1), y0 = y0,t_end=60)
main = RL_trainer(SP)

variance = 1
# for i in range(10):
    # variance += 1
main.train(variance = variance)
