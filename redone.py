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

    angle = 0 at straight down.
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

class SinglePendulum:
    def __init__(self, params = (9.81, 1, 1, 1), y0 = [0, np.pi, 0, 0], t_start = 0.0, t_end = 20, fps = 60, max_motor_force = 100):
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
        m_c, m1 = sm.symbols('m_c m1')  # Masses
        L1 = sm.symbols('L1')           # Lengths
        I1 = sm.symbols('I1')           # Moments of Inertia (e.g., 1/12 * m * L**2)

        # 2. Define Generalized Coordinates (Functions of time)
        x = sm.Function('x')(t)
        th1 = sm.Function('th1')(t)

        q = [x, th1]
        dq = [sm.diff(qi, t) for qi in q]
        ddq = [sm.diff(dqi, t) for dqi in dq]

        # Extract velocities for cleaner math below
        dx, dth1 = dq

        # 3. Define Center of Mass (CoM) Positions
        # Cart
        x_c = x
        y_c = 0

        # Rod 1 (CoM is halfway down L1)
        x1 = x + (L1 / 2) * sm.sin(th1)
        y1 = -(L1 / 2) * sm.cos(th1)

        # 4. Calculate Velocities (Time derivatives of positions)
        v_c_x = sm.diff(x_c, t)

        v1_x = sm.diff(x1, t)
        v1_y = sm.diff(y1, t)

        # 5. Define Energies
        # Kinetic Energy: T = T_trans_cart + (T_trans_1 + T_rot_1) + (T_trans_2 + T_rot_2)
        T_cart = 0.5 * m_c * v_c_x**2
        T1 = 0.5 * m1 * (v1_x**2 + v1_y**2) + 0.5 * I1 * dth1**2
        T = T_cart + T1

        # Potential Energy: V = m * g * y_com
        V = m1 * g * y1

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
        inputs = [g, m_c, m1, L1, I1, x, th1, dx, dth1]
        M_inputs = [m_c, m1, L1, I1, th1] # M only depends on positions and constants

        # These will compile almost instantly
        calc_M = sm.lambdify(M_inputs, M, "numpy")
        calc_F = sm.lambdify(inputs, F, "numpy")

        print("Ready")
        return calc_M, calc_F

    def get_accelerations(self, state):
        x, th1, dx, dth1 = state
        g, m_c, m1, L1 = self.params
        I1 = 1/3 * m1 * L1**2

        # Calculate physics matrices (from SymPy)
        M_num = self.calc_M(m_c, m1, L1, I1, th1)
        F_num = self.calc_F(g, m_c, m1, L1, I1, x, th1, dx, dth1).flatten()
        
        # Add your motor input! 
        # (Assuming index 0 is the cart's linear x-axis)
        tau = np.array([self.motor_force, 0.0])
        
        # Solve for accelerations
        ddq = np.linalg.solve(M_num, F_num + tau)
        
        # Return the first-order derivatives: [velocities, accelerations]
        return np.array([dx, dth1, ddq[0], ddq[1]])

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
        '''Just stabalize the cart at the desired location'''
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

    def solve_step_inverted_rod(self, target = 0, kp = 50, ki = 0, kd = 1600, mode = 'analog'):

        if mode == 'analog':
            angular_controller = pid.pid_controller(np.pi, self.state[1], kp, ki, kd)
            position_controller = pid.pid_controller(target, self.state[0], 13, 0, 1800, display = False)

        elif mode == 'RL':
            NN = nn.NeuralNetwork((6, 64, 64, 2), [nn.ReLU, nn.ReLU, [nn.linear, nn.sigmoid]], 'nn_library')
            NN.theta_recover()
            
        cost = 0
        offset = 0
        discount = 0.2
        stable_counter = 0 
        state_history = []

        for t in self.t_eval:
            if mode == 'analog':
                
                angular_controller.location = self.state[1]
                position_controller.location = self.state[0]
                position_input = position_controller.update()
                angular_input = angular_controller.update()

                # Stage 0: If in excessive motion, stablalize
                if abs(self.state[3]) > 10 or abs(self.state[2]) > 10 or stable_counter == -1:
                    state_string = 'excessive motion'
                    self.motor_force = -(self.state[0] - target + self.state[2]) * self.max_motor_force * 0.1
                    stable_counter = -1
                    if abs(self.state[3]) < np.pi/2 and abs(self.state[2]) < 2:
                        stable_counter = 0

                # Stage 1: Initialize swing
                elif abs(self.state[1]) < 0.01 and abs(self.state[3]) < 0.01:
                    state_string = 'initialize swing'
                    self.motor_force = 10
                    stable_counter = 0
                
                # Stage 2: increase amplitude
                elif self.state[1] <= np.pi/2 or self.state[1] >= 3 * np.pi/2:
                    state_string = 'increase amplitude'
                    if self.state[3] > 0:
                        self.motor_force = -self.max_motor_force * math.cos(self.state[1])
                    else:
                        self.motor_force = self.max_motor_force * math.cos(self.state[1])
                    self.motor_force *= discount
                    stable_counter = 0
                
                # Stage 3: Kick to inverted position
                elif self.state[1] >= np.pi/2 and self.state[1] <= 3 * np.pi/2 and abs(self.state[1] - np.pi + 0.2 * math.atan(offset)) >= np.pi/5:
                    state_string = 'kick to inverted position'
                    angular_controller.kp = 5
                    angular_controller.kd = 200
                    angular_controller.target = np.pi
                    self.motor_force = angular_input
                    stable_counter = 0

                # Stage 4: Maintain 
                else:
                    state_string = 'maintain'
                    angular_controller.kp = kp
                    angular_controller.kd = kd
                    angular_controller.target = np.pi

                    stable_counter += 1 
                    if stable_counter >= 0.5 * self.fps:
                        offset = -0.1 * position_input
                        angular_controller.target = np.pi + 0.02 * math.atan(offset)

                    self.motor_force = angular_input
                

            elif mode == 'RL':
                self.motor_force = NN.feedforward(self.state)[-1][0][1] * 100
            
            # reject if the motor is asked to do more than it could
            if abs(self.motor_force) >= self.max_motor_force:
                print('Overload at ', t, 's')
                if self.motor_force > 0:
                    self.motor_force = self.max_motor_force
                else:
                    self.motor_force = -self.max_motor_force

            # 2. Advance the physics by exactly one frame (dt)
            self.state = self.rk4_step()
            self.solution.append(np.append(self.state, self.motor_force))
            state_history.append(state_string)

            self.current_time += self.dt
            cost += (target - self.state[0])**2
            
        return np.array(self.solution), cost, state_history

    def animate(self, mode = 'RL', speed = 1):
        solution, cost, state_history = self.solve_step_inverted_rod()
        print('cost =', cost)
        # Extract position arrays for the animation
        x_cart_history = solution[:, 0]
        th1_history = solution[:, 1]
        force_history = solution[:, 4]

        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xlim(-10, 10)
        ax.set_ylim(-3, 7)

        # Setup empty artists
        cart_marker, = ax.plot([], [], 'ks', markersize=10) # Black square for cart
        rod1, = ax.plot([], [], 'b-', lw=2)
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
        force_text = ax.text(0.05, 0.6, '', transform=ax.transAxes, fontsize=12)
        state_text = ax.text(0.05, 0.5, '', transform=ax.transAxes, fontsize=12)

        def init():
            cart_marker.set_data([], [])
            rod1.set_data([], [])
            
            # Reset the arrow to zero length at the origin
            force_arrow.set_offsets([[0, 0]])
            force_arrow.set_UVC(0, 0)
            
            # 2. Clear the text in the initialization
            time_text.set_text('')
            location_text.set_text('')
            angle_1_text.set_text('')
            force_text.set_text('')
            state_text.set_text('')
            
            # 3. Return the text artist alongside the others, including force_arrow
            return cart_marker, rod1, force_arrow, time_text, location_text, angle_1_text, force_text, state_text

        def update(frame):
            x_c = x_cart_history[frame]
            th1 = th1_history[frame]
            f = force_history[frame]
            state = state_history[frame]
            
            x1 = x_c + 1.0 * np.sin(th1)
            y1 = -1.0 * np.cos(th1)
                        
            cart_marker.set_data([x_c], [0])
            rod1.set_data([x_c, x1], [0, y1])
            
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
            force_text.set_text(f'Force: {f:.2f} N')
            state_text.set_text(f'State: {state}')
            
            # 5. Return the updated text artist
            return cart_marker, rod1, force_arrow, time_text, location_text, angle_1_text, force_text, state_text


        ani = animation.FuncAnimation(
            fig, update, frames=len(self.t_eval), 
            init_func=init, blit=True, interval=1000/self.fps * speed
        )
        plt.show()

if __name__ == "__main__":
    SP = SinglePendulum(params=(9.8, 1, 1, 1), y0 = [0, 0, 0, 0],t_end=60)
    SP.animate()