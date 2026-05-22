# Frozen Lake - Reinforcement Learning Assignment

**Author:** Chris Hirschbrich

## Overview

Comparison of three reinforcement learning algorithms on the Gymnasium Frozen Lake environment - a 4×4 grid where an agent must reach the goal without falling through holes on slippery ice.

## Algorithms

- **MC with Exploring Starts** - from class materials
- **MC with ε-soft Policy** - implemented from scratch for this assignment
- **Q-Learning** - TD control with tunable hyperparameters

## Experiments

Each algorithm was tested across four environments with increasing slipperiness (100%, 90%, 70%, 50% success rate) over 20,000 episodes. A bonus experiment tested Q-Learning under different reward structures (hole penalty, step penalty, combined).

## Key Findings

- MC with ES converged fastest in deterministic environments but performed similarly to MC ε-soft once the environment became stochastic.
- Q-Learning struggled with sparse rewards but improved significantly with a step penalty, which provided a denser learning signal.
- In highly stochastic environments, the choice of exploration strategy matters less - slippery ice naturally diversifies the agent's experience.

## Setup

```bash
pip install gymnasium[box2d,classic_control]
git clone https://github.com/mhahsler/gym-classics.git
cd gym-classics && pip install -e .
```
