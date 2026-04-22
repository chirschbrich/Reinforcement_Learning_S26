# Value Function Approximation for the Lunar Lander Problem

**Author:** Chris Hirschbrich

## Overview

This project applies **tile coding** with semi-gradient SARSA(0) to learn a control policy for two reinforcement learning environments: the discrete L-Maze and the continuous-state Lunar Lander (Gymnasium `LunarLander-v3`).

## Tasks

### Task 1 — Tile Coding on the L-Maze
Implements a custom tile coding feature transform and uses it as a drop-in replacement for Fourier basis features. Hyperparameters (number of tilings, tile size, learning rate) are tuned via a grid search evaluated with root-mean-squared value error (MSVE) against the true value function from dynamic programming. Best result: **8 tilings, tile size 2.0**.

### Task 2 — Tile Coding on Lunar Lander
Scales the tile coding approach to the 8-dimensional continuous state space of Lunar Lander. Key design choices:
- 16 tilings over 6 continuous observations, with one-hot encoding of the two binary leg-contact sensors
- A hashed index table (IHT) of size 8192 to manage the feature space
- Custom reward shaping: bonus of +0.1 when the agent chooses NO_OP while safely landed, to prevent the lander from stalling indefinitely
- Trained for 20,000 episodes; achieved a **91% success rate** on 100 evaluation episodes

### Task 3 — Discussion
Covers the challenges of scaling tile coding from a 2D grid world to a high-dimensional continuous environment, including hyperparameter sensitivity (α, ε decay, tile count) and a discussion of why dense reward shaping outperforms sparse +100/−100 reward structures.

## Dependencies

```
numpy
gymnasium[box2d,other]
swig
pyvirtualdisplay
gym-classics (patched internal build)
matplotlib
```

## Usage

Open `lunar_lander_approximation.ipynb` in Jupyter or Google Colab and run all cells in order. The notebook installs its own dependencies and downloads required helper scripts automatically.
