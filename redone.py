import math
import random
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import numpy as np
import scipy
from scipy.integrate import solve_ivp
import sympy as sm
import pid 
import neural_network as nn

class DoublePendulum:
    """
    A class representing a Double Pendulum on a cart system.
    This class encapsulates the physics engine and numerical integration
    to simulate the dynamics of the system.
    """
    def __init__(self, params = (9.81, 1, 1, 1, 1, 1), y0 = [0, np.pi, np.pi/6, 0, 0, 0], t_start = 0.0, t_end = 20, fps = 60, max_motor_force = 100):
        self.g = 9.81
        self.calc_M, self.calc_F = self.physics_engine()
        self.params = params
        self.state = y0
        self.motor_force = 0
        self.t_start = t_start
        self.t_end = t_end      # Simulate 10 seconds
        self.fps = fps          # Animation frames per second
        self.current_time = t_start
        self.t_eval = np.linspace(t_start, t_end, int((t_end - t_start) * fps))
        self.dt = self.t_eval[1] - self.t_eval[0]
        self.solution = []
        self.max_motor_force = max_motor_force

    def physics_engine(self):
        # 1. Define Time and Constants
        t = sm.Symbol('t')
        g = sm.Symbol('g')
        m_c, m1, m2 = sm.symbols('m_c m1 m2')  # Masses
        L1, L2 = sm.symbols('L1 L2')           # Lengths
        I1, I2 = sm.symbols('I1 I2')           # Moments of Inertia (e.g., 1/12 * m * L**2)

        # 2. Define Generalized Coordinates (Functions of time)
        x = sm.Function('x')(t)
        th1 = sm.Function('th1')(t)
        th2 = sm.Function('th2')(t)

        q = [x, th1, th2]
        dq = [sm.diff(qi, t) for qi in q]
        ddq = [sm.diff(dqi, t) for dqi in dq]

        # Extract velocities for cleaner math below
        dx, dth1, dth2 = dq

        # 3. Define Center of Mass (CoM) Positions
        # Cart
        x_c = x
        y_c = 0

        # Rod 1 (CoM is halfway down L1)
        x1 = x + (L1 / 2) * sm.sin(th1)
        y1 = -(L1 / 2) * sm.cos(th1)

        # Rod 2 (Attached to end of L1, CoM is halfway down L2)
        x2 = x + L1 * sm.sin(th1) + (L2 / 2) * sm.sin(th2)
        y2 = -L1 * sm.cos(th1) - (L2 / 2) * sm.cos(th2)

        # 4. Calculate Velocities (Time derivatives of positions)
        v_c_x = sm.diff(x_c, t)

        v1_x = sm.diff(x1, t)
        v1_y = sm.diff(y1, t)

        v2_x = sm.diff(x2, t)
        v2_y = sm.diff(y2, t)

        # 5. Define Energies
        # Kinetic Energy: T = T_trans_cart + (T_trans_1 + T_rot_1) + (T_trans_2 + T_rot_2)
        T_cart = 0.5 * m_c * v_c_x**2
        T1 = 0.5 * m1 * (v1_x**2 + v1_y**2) + 0.5 * I1 * dth1**2
        T2 = 0.5 * m2 * (v2_x**2 + v2_y**2) + 0.5 * I2 * dth2**2
        T = T_cart + T1 + T2

        # Potential Energy: V = m * g * y_com
        V = m1 * g * y1 + m2 * g * y2

        # 6. Build the Lagrangian
        L = T - V

        # 7. Apply Euler-Lagrange
        equations = []
        for i, qi in enumerate(q):
            dL_ddq = sm.diff(L, dq[i])
            term1 = sm.diff(dL_ddq, t)
            term2 = sm.diff(L, qi)
            equations.append(term1 - term2)

        # 8. Extract the Mass Matrix (M) and the Forcing Vector (F)
        eq_matrix = sm.Matrix(equations)

        # The Jacobian of the equations with respect to the accelerations IS the Mass matrix
        M = eq_matrix.jacobian(ddq)

        # The rest of the terms (Coriolis, Gravity) are found by setting accelerations to 0
        F = -eq_matrix.subs({ddq_i: 0 for ddq_i in ddq})

        # 9. Compile M and F into fast NumPy functions
        # Notice we no longer need dx, dth1, dth2 for the Mass matrix
        inputs = [g, m_c, m1, m2, L1, L2, I1, I2, x, th1, th2, dx, dth1, dth2]
        M_inputs = [m_c, m1, m2, L1, L2, I1, I2, th1, th2] # M only depends on positions and constants

        # These will compile almost instantly
        calc_M = sm.lambdify(M_inputs, M, "numpy")
        calc_F = sm.lambdify(inputs, F, "numpy")

        print("Ready")
        return calc_M, calc_F

    def get_accelerations(self, state):
        x, th1, th2, dx, dth1, dth2 = state
        g, m_c, m1, m2, L1, L2 = self.params
        I1 = 1/3 * m1 * L1**2
        I2 = 1/3 * m2 * L2**2

        
        # Calculate physics matrices (from SymPy)
        M_num = self.calc_M(m_c, m1, m2, L1, L2, I1, I2, th1, th2)
        F_num = self.calc_F(g, m_c, m1, m2, L1, L2, I1, I2, x, th1, th2, dx, dth1, dth2).flatten()
        
        # Add your motor input! 
        # (Assuming index 0 is the cart's linear x-axis)
        tau = np.array([self.motor_force, 0.0, 0.0])
        
        # Solve for accelerations
        ddq = np.linalg.solve(M_num, F_num + tau)
        
        # Return the first-order derivatives: [velocities, accelerations]
        return np.array([dx, dth1, dth2, ddq[0], ddq[1], ddq[2]])

    def rk4_step(self):
        """Calculates the next state of the system using RK4 integration."""
        
        # k1: Slope at the beginning of the interval
        k1 = self.get_accelerations(self.state)
        
        # k2: Slope at the midpoint (using k1)
        k2 = self.get_accelerations(self.state + 0.5 * self.dt * k1)
        
        # k3: Slope at the midpoint (using k2)
        k3 = self.get_accelerations(self.state + 0.5 * self.dt * k2)
        
        # k4: Slope at the end of the interval (using k3)
        k4 = self.get_accelerations(self.state + self.dt * k3)
        
        # Weighted average of the slopes yields the new state
        new_state = self.state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        while new_state[1] > 2 * np.pi:
            new_state[1] -= 2*np.pi
        while new_state[1] < 0:
            new_state[1] += 2*np.pi
        while new_state[2] > 2 * np.pi:
            new_state[2] -= 2*np.pi
        while new_state[2] < 0:
            new_state[2] += 2*np.pi
        
        return new_state

    def solve_step_null_input(self):

        for t in self.t_eval:

            # 1. Decide on your motor torque for this specific frame
            self.motor_force = 0

            # 2. Advance the physics by exactly one frame (dt)
            self.state = self.rk4_step()
            self.solution.append(np.append(self.state, 0))
            self.current_time += self.dt
            
        return np.array(self.solution), 0

    def solve_step_stablize_position(self, target = 0, kp = 4, ki = 0.001, kd = 400):

        controller = pid.pid_controller(target, self.state[0], kp, ki, kd)
        cost = 0

        for t in self.t_eval:

            # 1. Decide on your motor torque for this specific frame
            controller.location = self.state[0]
            if t >= 2:
                self.motor_force = min((controller.update(), self.max_motor_force), key=abs)
            else:
                self.motor_force = 0
            # 2. Advance the physics by exactly one frame (dt)
            self.state = self.rk4_step()
            self.solution.append(np.append(self.state, self.motor_force))
            self.current_time += self.dt
            cost += (target - self.state[0])**2
            
        return np.array(self.solution), cost

    def solve_step_inverted_rod_1(self, kp = 15, ki = 0.1, kd = 2000, mode = 'analog'):

        if mode == 'analog':
            controller = pid.pid_controller(np.pi, self.state[1], kp, ki, kd)
        elif mode == 'RL':
            NN = nn.NeuralNetwork((6, 64, 64, 2), [nn.ReLU, nn.ReLU, [nn.linear, nn.sigmoid]], 'nn_library')
            NN.theta_recover()
            
        cost = 0
        for t in self.t_eval:
            if mode == 'analog':
                # Stage 1: Initialize swing
                if self.state[1] == 0 and self.state[4] == 0:
                    self.motor_force = 100
                
                # Stage 2: increase amplitude
                elif self.state[1] <= np.pi/2 or self.state[1] >= 3 * np.pi/2:
                    # print('increase')
                    if self.state[4] > 0:
                        self.motor_force = -self.max_motor_force * math.cos(self.state[1])
                    else:
                        self.motor_force = self.max_motor_force * math.cos(self.state[1])
                
                # Stage 3: Kick to inverted position
                elif self.state[1] >= np.pi/2 and self.state[1] <= 3 * np.pi/2 and abs(self.state[1] - np.pi) >= np.pi/10:
                    controller.kp =5
                    controller.kd = 100
                    controller.location = self.state[1]
                    self.motor_force = controller.update()

                # Stage 4: Maintain 
                else:
                    controller.kp = kp
                    controller.kd = kd
                    controller.location = self.state[1]
                    self.motor_force = controller.update()
            elif mode == 'RL':
                self.motor_force = NN.feedforward(self.state)[-1][0][1] * 100
            
            # reject if the motor is asked to do more than it could
            if abs(self.motor_force) >= self.max_motor_force:
                print('Overload at ', t, 's')
            self.motor_force = min((self.motor_force, self.max_motor_force), key=abs)

            # 2. Advance the physics by exactly one frame (dt)
            self.state = self.rk4_step()
            self.solution.append(np.append(self.state, self.motor_force))
            self.current_time += self.dt
            cost += (np.pi - self.state[1])**2 + (np.pi - self.state[2])**2
            
        return np.array(self.solution), cost

    def animate(self):
        solution, cost = self.solve_step_inverted_rod_1()
        print('cost =', cost)
        # Extract position arrays for the animation
        x_cart_history = solution[:, 0]
        th1_history = solution[:, 1]
        th2_history = solution[:, 2]
        force_history = solution[:, 6]

        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xlim(-5, 15)
        ax.set_ylim(-3, 7)

        # Setup empty artists
        cart_marker, = ax.plot([], [], 'ks', markersize=10) # Black square for cart
        rod1, = ax.plot([], [], 'b-', lw=2)
        rod2, = ax.plot([], [], 'r-', lw=2)
        rail = ax.plot([-30,30], [0, 0], 'k-', lw=2)

        # --- NEW: Setup the quiver object for the force arrow ---
        # scale_units='xy' and scale=1 means the arrow length matches plot coordinates.
        # We will manually scale the force magnitude below to keep it visually manageable.
        force_arrow = ax.quiver([0], [0], [0], [0], color='green', pivot='tail', 
                                angles='xy', scale_units='xy', scale=1, width=0.01, zorder=5)
        force_scale = 0.1 # Adjust this multiplier to change how long the arrow draws

        # 1. Setup the empty text object for the time
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes, fontsize=12)
        location_text = ax.text(0.05, 0.8, '', transform=ax.transAxes, fontsize=12)
        angle_1_text = ax.text(0.05, 0.7, '', transform=ax.transAxes, fontsize=12)
        angle_2_text = ax.text(0.05, 0.6, '', transform=ax.transAxes, fontsize=12)
        force_text = ax.text(0.05, 0.5, '', transform=ax.transAxes, fontsize=12)

        def init():
            cart_marker.set_data([], [])
            rod1.set_data([], [])
            rod2.set_data([], [])
            
            # Reset the arrow to zero length at the origin
            force_arrow.set_offsets([[0, 0]])
            force_arrow.set_UVC(0, 0)
            
            # 2. Clear the text in the initialization
            time_text.set_text('')
            location_text.set_text('')
            angle_1_text.set_text('')
            angle_2_text.set_text('')
            force_text.set_text('')
            
            # 3. Return the text artist alongside the others, including force_arrow
            return cart_marker, rod1, rod2, force_arrow, time_text, location_text, angle_1_text, angle_2_text, force_text

        def update(frame):
            x_c = x_cart_history[frame]
            th1 = th1_history[frame]
            th2 = th2_history[frame]
            f = force_history[frame]
            
            x1 = x_c + 1.0 * np.sin(th1)
            y1 = -1.0 * np.cos(th1)
            
            x2 = x1 + 1.0 * np.sin(th2)
            y2 = y1 - 1.0 * np.cos(th2)
            
            cart_marker.set_data([x_c], [0])
            rod1.set_data([x_c, x1], [0, y1])
            rod2.set_data([x1, x2], [y1, y2])
            
            # --- NEW: Update the force arrow ---
            # set_offsets sets the x,y starting coordinate of the arrow
            force_arrow.set_offsets([[x_c, 0]])
            # set_UVC sets the dx, dy vector components of the arrow
            force_arrow.set_UVC(f * force_scale, 0)
            
            # 4. Update the text string using the t_eval array
            current_time = self.t_eval[frame]
            time_text.set_text(f'Time: {current_time:.2f} s')
            location_text.set_text(f'Location: {x_c:.2f} m')
            angle_1_text.set_text(f'Angle 1: {th1:.2f} rad')
            angle_2_text.set_text(f'Angle 2: {th2:.2f} rad')
            force_text.set_text(f'Force: {f:.2f} N')
            
            # 5. Return the updated text artist
            return cart_marker, rod1, rod2, force_arrow, time_text, location_text, angle_1_text, angle_2_text, force_text


        ani = animation.FuncAnimation(
            fig, update, frames=len(self.t_eval), 
            init_func=init, blit=True, interval=1000/self.fps
        )
        plt.show()


DP = DoublePendulum()
DP.animate()


class Auto_Damper(DoublePendulum):
    def __init__(self, max_motor_force, dt,params = (9.81, 1, 1, 1, 1, 1)):
        self.NN = nn.NeuralNetwork((6, 64, 64, 2), [nn.ReLU, nn.ReLU, nn.sigmoid], 'nn_library')
        self.params = params
        self.log_std = 2.5
        self.d_log_std = 0
        self.max_motor_force = max_motor_force
        self.dt = dt
        # self.NN.theta_generate()
        self.X = self.NN.theta_recover()


    def critic(self, state, action_force):
        return  1 - 2 *  math.cos(state[1]) - state[0]**2

    def backward_std(self, action, mu, sigma, advantage):
        # 1. Calculate the gradient for the mean
        # This acts as the "dZ" for your output layer, which you will backprop 
        # through W_out_mean and your hidden layers.
        d_mu = -advantage * ((action - mu) / (sigma ** 2))
        
        # 2. Calculate the gradient for the standalone variable
        # We accumulate this gradient directly
        self.d_log_std += -advantage * (((action - mu)**2 / (sigma ** 2)) - 1.0)
        
        # ... proceed to backprop d_mu through the rest of the MLP ...
        return d_mu
    
    def train(self):
        gamma = 0.99  # Discount factor (how much we care about the future)
        learning_rate = 0.001 # For your custom optimizer
        reward_history = []
        log_std_history = []
        # We train over "Episodes", not Epochs. 
        # An episode is one attempt at balancing until it falls.
        for episode in range(60000):
            # 1. Reset the simulation to the starting position (e.g., straight up)
            done = False
            total_episode_reward = 0
            t = 0
            d_V_cumi = 0
            d_mu_cumi = 0
            self.d_log_std = 0

            while not done:
                t += 1
                # --- THE FORWARD PASS ---
                # The brain looks at the state and picks a force
                # nn[0] = V, nn[1] = mu (Actual force)
                NN_output = self.NN.feedforward(state)[-1][0]
                self.motor_force = NN_output[1] + np.exp(self.log_std) * np.random.randn()
                current_value = NN_output[0]
                           
            
                # --- THE PHYSICS ENGINE ---
                # The cart moves for 0.02 seconds using the chosen force
                next_state = self.rk4_step(
                                state,        # y
                                self.params,               # *args: gravity, masses, lengths
                                action_force,           # *args: dynamic inputs
                                self.dt)
                reward = self.critic(next_state, action_force)
                total_episode_reward += reward
                done = next_state[1] <= np.pi/2 or next_state[1] >= 3*np.pi/2 or t >= 2000
                    

                # --- THE TARGET CALCULATION ---
                # What is the value of the state we just landed in?
                if done:
                    target_value = reward # If we died, there is no future.
                else:
                    next_value = self.NN.feedforward(next_state)[-1][0][1] + np.exp(self.log_std) * np.random.randn()
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
                    sigma=np.exp(self.log_std), 
                    advantage=advantage
                )
                if d_V > 0.5 or d_mu > 0.5:
                    d_V /= max(d_V, d_mu)
                    d_mu /= max(d_V, d_mu)
                
                d_V_cumi += d_V
                d_mu_cumi += d_mu

                # 3. Route the gradients backward through the shared trunk
                # You will need to write a backward_trunk(d_mu, d_V) function that 
                # applies the chain rule back through W_mu, W_v, W2, W1, and the ReLUs!
            
                

                # self.d_log_std = 0.0 # Reset accumulator for the next step
                
                # Move to the next frame
                state = next_state
            

            self.NN.backward(np.array([state]), (action_force + d_mu/t, current_value + d_V/t), learning_rate)
            self.log_std -= learning_rate * self.d_log_std/t
            self.log_std = np.clip(self.log_std, 0, 3.0)


            if episode % 100 == 0:
                print(f"Episode {episode} finished! Total Reward: {total_episode_reward}")
                self.NN.theta_save()
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

if __name__ == "__main__":
    main = Auto_Damper([0, np.pi, np.pi/6, 0, 0, 0], 100, 1/60)
    for i in range(10):
        main.train()
