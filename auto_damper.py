import slider
import neural_network as nn
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp
import redone

class Auto_Damper():
    def __init__(self, state, max_motor_force, dt,params = (9.81, 1, 1, 1, 1, 1)):
        self.NN = nn.NeuralNetwork((6, 64, 64, 2), [nn.ReLU, nn.ReLU, [nn.linear, nn.sigmoid]], 'nn_library')
        self.x, self.th1, self.th2, self.dx, self.dth1, self.dth2 = state
        self.params = params
        self.log_std = 2.5
        self.d_log_std = 0
        self.max_motor_force = max_motor_force
        self.dt = dt

        self.NN.theta_generate()

        self.X = self.NN.theta_recover()

    def critic(self, state):
        location_cf = 10
        return  (15 - 2 *  math.cos(state[1]) - location_cf * state[0]**2) / 100

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
     
    def train(self):

        gamma = 0.99  # Discount factor (how much we care about the future)
        learning_rate = 0.001 # For your custom optimizer
        reward_history = []
        log_std_history = []

        for episode in range(500):
            # Reset the simulation to the starting position
            state = np.array([self.x, self.th1, self.th2, self.dx, self.dth1, self.dth2])
            scale_factors = np.array([10.0, 3.14, 3.14, 50.0, 50.0, 50.0])

            done = False
            total_episode_reward = 0
            t = 0
            d_V_cumi = 0
            d_mu_cumi = 0
            self.d_log_std = 0
            states_memory = []
            targets_memory = []

            while not done:
                t += 1
 
                # nn[0] = V (Score), nn[1] = mu (Actual force)
                # Normalize before asking for an action
                normalized_state = state / scale_factors
                NN_output = self.NN.feedforward(normalized_state)[-1][0]
                action_force = (NN_output[1] - 0.5) * 200 + np.exp(self.log_std) * np.random.randn()
                current_value = NN_output[0]
                           
                # --- THE PHYSICS ENGINE ---
                # The cart moves for 0.02 seconds using the chosen force
                next_state = slider.rk4_step(
                                state,        # y
                                self.params,               # *args: gravity, masses, lengths
                                action_force,           # *args: dynamic inputs
                                self.dt)
                reward = self.critic(next_state)
                total_episode_reward += reward
                has_nan = np.isnan(next_state).any() 
                done = has_nan or next_state[1] <= np.pi/2 or next_state[1] >= 3*np.pi/2 or t >= 2000 or abs(next_state[3]) > 100 or abs(next_state[4]) > 100 or abs(next_state[0]) > 100 or abs(next_state[5]) > 100                

                # --- THE TARGET CALCULATION ---
                # Value of the state we just landed in
                if done:
                    target_value = reward # If we died, there is no future.
                else:
                    normalized_next_state = next_state / scale_factors
                    next_value = self.NN.feedforward(normalized_next_state)[-1][0][0]
                    target_value = reward + gamma * next_value
                    
                # Advantage: Was the move better than the Critic expected?
                advantage = target_value - current_value
                
                # --- THE BACKWARD PASS ---
                # 1. Backprop for the Critic (Mean Squared Error)
                # Loss = 0.5 * (target_value - current_value)^2
                # dL/dV = -(target_value - current_value) = -advantage
                d_V = -advantage
                
                # 2. Backprop for the Actor
                # This function updates self.d_log_std internally and returns the gradient for mu
                d_mu = self.backward_std(
                    action=action_force, 
                    mu=NN_output[1], 
                    sigma=np.exp(self.log_std)/200, 
                    advantage=advantage)

                # Capping function
                if abs(d_V) > 1.0: d_V = np.sign(d_V) * 1.0
                if abs(d_mu) > 1.0: d_mu = np.sign(d_mu) * 1.0                
                
                d_V_cumi += d_V
                d_mu_cumi += d_mu
                
                # Move to the next frame
                state = next_state

                states_memory.append(normalized_state)
                target_V = current_value - d_V
                target_mu = NN_output[1] - d_mu
                targets_memory.append([target_V, target_mu])


            self.NN.backward(np.array(states_memory), np.array(targets_memory), learning_rate/t)
            self.log_std -= learning_rate * self.d_log_std/t
            self.log_std = np.clip(self.log_std, 0, 3.0)
            if t>= 1000:
                print('runtime = ', t)
            if total_episode_reward >= 2500:
                print(episode)
                DP = redone.DoublePendulum()
                DP.animate()
            print(f"Episode {episode} finished! Total Reward: {total_episode_reward}")

            if episode % 100 == 0:
                self.NN.theta_save()
                print('Saved!')
            reward_history.append(total_episode_reward)
            log_std_history.append(self.log_std)
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(reward_history)
        ax2.plot(log_std_history)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Log Std')
        plt.show()
        DP = redone.DoublePendulum()
        DP.animate()


main = Auto_Damper([0, np.pi, np.pi/6, 0, 0, 0], 100, 1/60)
# DP = redone.DoublePendulum()
# DP.animate()
for i in range(10):
    main.train()