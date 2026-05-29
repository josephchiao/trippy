import math
import random
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import numpy as np
import scipy
from scipy.integrate import solve_ivp
import sympy as sm
import pid 


def physics_engine():
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

calc_M, calc_F = physics_engine()

def get_accelerations(state, params, motor_force):
    x, th1, th2, dx, dth1, dth2 = state
    g, m_c, m1, m2, L1, L2 = params
    I1 = 1/3 * m1 * L1**2
    I2 = 1/3 * m2 * L2**2

    
    # Calculate physics matrices (from SymPy)
    M_num = calc_M(m_c, m1, m2, L1, L2, I1, I2, th1, th2)
    F_num = calc_F(g, m_c, m1, m2, L1, L2, I1, I2, x, th1, th2, dx, dth1, dth2).flatten()
    
    # Add your motor input! 
    # (Assuming index 0 is the cart's linear x-axis)
    tau = np.array([motor_force, 0.0, 0.0])
    
    # Solve for accelerations
    ddq = np.linalg.solve(M_num, F_num + tau)
    
    # Return the first-order derivatives: [velocities, accelerations]
    return np.array([dx, dth1, dth2, ddq[0], ddq[1], ddq[2]])

def rk4_step(state, params, motor_force, dt):
    """Calculates the next state of the system using RK4 integration."""
    
    # k1: Slope at the beginning of the interval
    k1 = get_accelerations(state, params, motor_force)
    
    # k2: Slope at the midpoint (using k1)
    k2 = get_accelerations(state + 0.5 * dt * k1, params, motor_force)
    
    # k3: Slope at the midpoint (using k2)
    k3 = get_accelerations(state + 0.5 * dt * k2, params, motor_force)
    
    # k4: Slope at the end of the interval (using k3)
    k4 = get_accelerations(state + dt * k3, params, motor_force)
    
    # Weighted average of the slopes yields the new state
    new_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    while new_state[1] > 2 * np.pi:
        new_state[1] -= 2*np.pi
    while new_state[1] < 0:
        new_state[1] += 2*np.pi
    while new_state[2] > 2 * np.pi:
        new_state[2] -= 2*np.pi
    while new_state[2] < 0:
        new_state[2] += 2*np.pi
    
    return new_state

# 1. Define Initial Conditions
# Starts at x=0, theta1=90deg, theta2=90deg, all velocities = 0

params = (9.81, 1, 1, 1, 1, 1)


# For stablize position test:
# y0 = [-7.5, 1, -1, 0, 0, 0]

# For inverted rod one test:
y0 = [0, np.pi, np.pi/6, 0, 0, 0]
# y0 = [0, 0, 0, 0, 0, 0]



# 2. Define Time Span and Evaluation Points
t_start = 0.0
t_end = 20      # Simulate 10 seconds
fps = 60          # Animation frames per second
t_eval = np.linspace(t_start, t_end, int((t_end - t_start) * fps))

def solve_step():
    current_time = t_start
    current_state = y0
    dt = t_eval[1] - t_eval[0]
    solution = []

    for t in t_eval:

        # 1. Decide on your motor torque for this specific frame
        motor_force = 0

        # 2. Advance the physics by exactly one frame (dt)
        current_state = rk4_step(
            current_state,        # y
            params,               # *args: gravity, masses, lengths
            motor_force,           # *args: dynamic inputs
            dt
        )
        solution.append(current_state)
        current_time += dt
        
    return np.array(solution)

def solve_step_stablize_position(target = 0, kp = 4, ki = 0.001, kd = 400, max_motor_force= 100):
    current_time = t_start
    current_state = y0
    dt = t_eval[1] - t_eval[0]
    solution = []
    controller = pid.pid_controller(target, current_state[0], kp, ki, kd)
    cost = 0
    for t in t_eval:

        # 1. Decide on your motor torque for this specific frame
        controller.location = current_state[0]
        if t >= 2:
            motor_force = min((controller.update(), max_motor_force), key=abs)
        else:
            motor_force = 0
        # 2. Advance the physics by exactly one frame (dt)
        current_state = rk4_step(
            current_state,        # y
            params,               # *args: gravity, masses, lengths
            motor_force,           # *args: dynamic inputs
            dt
        )
        solution.append(np.append(current_state, motor_force))
        current_time += dt
        cost += (target - current_state[0])**2
        
    return np.array(solution), cost


def solve_step_inverted_rod_1(kp = 15, ki = 0.1, kd = 2000, max_motor_force= 100, mode = 'analog'):
    current_time = t_start
    current_state = y0
    dt = t_eval[1] - t_eval[0]
    solution = []
    controller = pid.pid_controller(np.pi, current_state[1], kp, ki, kd)
    cost = 0
    for t in t_eval:
        if mode == 'analog':
            # Stage 1: Initialize swing
            if current_state[1] == 0 and current_state[4] == 0:
                motor_force = 100
            
            # Stage 2: increase amplitude
            elif current_state[1] <= np.pi/2 or current_state[1] >= 3 * np.pi/2:
                # print('increase')
                if current_state[4] > 0:
                    motor_force = -max_motor_force * math.cos(current_state[1])
                else:
                    motor_force = max_motor_force * math.cos(current_state[1])
            
            # Stage 3: Kick to inverted position
            elif current_state[1] >= np.pi/2 and current_state[1] <= 3 * np.pi/2 and abs(current_state[1] - np.pi) >= np.pi/10:
                controller.kp =5
                controller.kd = 100
                controller.location = current_state[1]
                motor_force = controller.update()

            # Stage 4: Maintain 
            else:
                controller.kp = kp
                controller.kd = kd
                controller.location = current_state[1]
                motor_force = controller.update()
        elif mode == 'RL':
            motor_force = 0
        

        # reject if the motor is asked to do more than it could
        if abs(motor_force) >= max_motor_force:
            print('Overload at ', t, 's')
        motor_force = min((motor_force, max_motor_force), key=abs)

        # 2. Advance the physics by exactly one frame (dt)
        current_state = rk4_step(
            current_state,        # y
            params,               # *args: gravity, masses, lengths
            motor_force,           # *args: dynamic inputs
            dt
        )
        solution.append(np.append(current_state, motor_force))
        current_time += dt
        cost += (np.pi - current_state[1])**2 + (np.pi - current_state[2])**2
        
    return np.array(solution), cost

# region SOLVE IVP
# solution = solve_ivp(
#     fun=pendulum_derivatives,
#     t_span=(t_start, t_end),
#     y0=y0,
#     t_eval=t_eval,       # Forces solver to output data exactly at our animation frames
#     args=(params,),      # Passes our constants to the derivative function
#     method='RK45',       # Explicit Runge-Kutta. 
#     rtol=1e-6,           # Relative tolerance (decrease if energy isn't conserved)
#     atol=1e-8            # Absolute tolerance
# )

# if not solution.success:
#     print(f"Integration failed: {solution.message}")

# Extract position arrays for the animation
# x_cart_history = solution.y[0, :]
# th1_history = solution.y[1, :]
# th2_history = solution.y[2, :]
# endregion


# # region animate
# solution, cost = solve_step_inverted_rod_1()
# print(cost)
# # Extract position arrays for the animation
# # Extract position arrays for the animation
# x_cart_history = solution[:, 0]
# th1_history = solution[:, 1]
# th2_history = solution[:, 2]
# force_history = solution[:, 6]

# fig, ax = plt.subplots()
# ax.set_aspect('equal')
# ax.set_xlim(-5, 15)
# ax.set_ylim(-3, 7)

# # Setup empty artists
# cart_marker, = ax.plot([], [], 'ks', markersize=10) # Black square for cart
# rod1, = ax.plot([], [], 'b-', lw=2)
# rod2, = ax.plot([], [], 'r-', lw=2)
# rail = ax.plot([-30,30], [0, 0], 'k-', lw=2)

# # --- NEW: Setup the quiver object for the force arrow ---
# # scale_units='xy' and scale=1 means the arrow length matches plot coordinates.
# # We will manually scale the force magnitude below to keep it visually manageable.
# force_arrow = ax.quiver([0], [0], [0], [0], color='green', pivot='tail', 
#                         angles='xy', scale_units='xy', scale=1, width=0.01, zorder=5)
# force_scale = 0.1 # Adjust this multiplier to change how long the arrow draws

# # 1. Setup the empty text object for the time
# time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes, fontsize=12)
# location_text = ax.text(0.05, 0.8, '', transform=ax.transAxes, fontsize=12)
# angle_1_text = ax.text(0.05, 0.7, '', transform=ax.transAxes, fontsize=12)
# angle_2_text = ax.text(0.05, 0.6, '', transform=ax.transAxes, fontsize=12)
# force_text = ax.text(0.05, 0.5, '', transform=ax.transAxes, fontsize=12)

# def init():
#     cart_marker.set_data([], [])
#     rod1.set_data([], [])
#     rod2.set_data([], [])
    
#     # Reset the arrow to zero length at the origin
#     force_arrow.set_offsets([[0, 0]])
#     force_arrow.set_UVC(0, 0)
    
#     # 2. Clear the text in the initialization
#     time_text.set_text('')
#     location_text.set_text('')
#     angle_1_text.set_text('')
#     angle_2_text.set_text('')
#     force_text.set_text('')
    
#     # 3. Return the text artist alongside the others, including force_arrow
#     return cart_marker, rod1, rod2, force_arrow, time_text, location_text, angle_1_text, angle_2_text, force_text

# def update(frame):
#     x_c = x_cart_history[frame]
#     th1 = th1_history[frame]
#     th2 = th2_history[frame]
#     f = force_history[frame]
    
#     x1 = x_c + 1.0 * np.sin(th1)
#     y1 = -1.0 * np.cos(th1)
    
#     x2 = x1 + 1.0 * np.sin(th2)
#     y2 = y1 - 1.0 * np.cos(th2)
    
#     cart_marker.set_data([x_c], [0])
#     rod1.set_data([x_c, x1], [0, y1])
#     rod2.set_data([x1, x2], [y1, y2])
    
#     # --- NEW: Update the force arrow ---
#     # set_offsets sets the x,y starting coordinate of the arrow
#     force_arrow.set_offsets([[x_c, 0]])
#     # set_UVC sets the dx, dy vector components of the arrow
#     force_arrow.set_UVC(f * force_scale, 0)
    
#     # 4. Update the text string using the t_eval array
#     current_time = t_eval[frame]
#     time_text.set_text(f'Time: {current_time:.2f} s')
#     location_text.set_text(f'Location: {x_c:.2f} m')
#     angle_1_text.set_text(f'Angle 1: {th1:.2f} rad')
#     angle_2_text.set_text(f'Angle 2: {th2:.2f} rad')
#     force_text.set_text(f'Force: {f:.2f} N')
    
#     # 5. Return the updated text artist
#     return cart_marker, rod1, rod2, force_arrow, time_text, location_text, angle_1_text, angle_2_text, force_text


# ani = animation.FuncAnimation(
#     fig, update, frames=len(t_eval), 
#     init_func=init, blit=True, interval=1000/fps
# )
# # plt.show()
# endregion