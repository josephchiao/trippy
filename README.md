# trippy

Simulating and balancing an **inverted pendulum on a cart**, using classical (PID) control and a from-scratch neural-network controller. This is the original version of the project; a refactored second iteration lives in [`trippy_second`](https://github.com/josephchiao/trippy_second).

## What it does

A cart slides along a rail with one or two rods attached on top. Left alone the rods swing down; the goal is to drive the cart's motor so they balance upright. The project includes:

- **Physics engine** — cart + double-pendulum equations of motion derived symbolically with SymPy and integrated with SciPy's `solve_ivp` (see `redone.py`).
- **Controllers** — a PID controller ("analog" control) and a reinforcement-learning-trained neural network.
- **Neural network** — a small feed-forward network written from scratch in NumPy (`neural_network.py`), with saved weights in `nn_library/` and `nn_backup/`.
- **RL training** — `RL_training.py` trains the network to keep the pendulum upright and the cart centered.
- **Visualization** — Matplotlib animations of the cart and rods.

Current status: single-pendulum balancing works, and double-pendulum balancing works in two separate modes under analog control. A screen recording of a run is included in the repo.

## Repository layout

| File | Purpose |
|------|---------|
| `redone.py` | `SinglePendulum` / `DoublePendulum` classes — dynamics, integration, and animation (main entry point) |
| `neural_network.py` | From-scratch feed-forward neural network (NumPy) |
| `RL_training.py` | Reinforcement-learning trainer for the NN controller |
| `pid.py` | PID controller implementation |
| `auto_damper.py`, `motor.py`, `slider.py` | Supporting simulation components |
| `theta_init.py` | Weight initialization helpers |
| `nn_library/`, `nn_backup/` | Saved network weights (`.npz`) |

## Requirements

- Python 3.10+
- `numpy`, `scipy`, `sympy`, `matplotlib`

```bash
pip install numpy scipy sympy matplotlib
```

## Running it

```bash
python redone.py
```

By default this runs the single pendulum in reinforcement-learning (`RL`) balancing mode and shows the animation. Edit the `__main__` block at the bottom of `redone.py` to try the double pendulum or other initial conditions.

## Notes

This is a personal research/learning project and a work in progress.
Total work in progress. Give me some time
5/11 update: Single pendulum balancing, and double pendulums balancing in two seprate modes achieved with analog control.
