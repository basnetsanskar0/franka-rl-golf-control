# Autonomous Robotic Arm Control via Reinforcement Learning

This project trains a 7-DOF Franka Emika robotic arm to play a mini-golf game using deep reinforcement learning.

It was developed as part of the ArenaX Labs Machine Learning Hiring Challenge — where the objective is to teach a robot to:
- Grasp a golf club
- Align and swing with precision
- Hit a ball into a hole
- Avoid dropping the club or overshooting the hole

---

## Approach Overview

This repository includes two complete reinforcement learning pipelines, each valuing different strengths:

### 1. `train_golf_bot.py` — Stable-Baselines3 PPO (High Reliability)

Uses Stable-Baselines3 to quickly train an agent with a custom neural network.

- Quick to prototype and easy to run
- Leverages a proven PPO implementation
- Produces a submission-ready `.zip` model
- Suitable as a baseline for fast experimentation

### 2. `franka_golf_training.py` — Custom PyTorch PPO (High Flexibility)

Implements the full PPO algorithm from scratch, including:

- Custom actor-critic network with learnable log standard deviation
- Curriculum reward shaping: grasp → align → swing → precision
- Success tracking, GAE, and manual PPO updates
- Checkpoint saving, logging, and evaluation pipeline

This approach provides full transparency and control over learning behavior, ideal for fine-tuning or research-style development.

---

## Environment

- `FrankaIkGolfCourseEnv-v0` via `sai-rl`
- 7-dimensional continuous action space:  
  `[Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]`
- 31-dimensional observation space:  
  Includes joint states, ball, club, and hole positions and orientation

---

## Results

- Trained robot learns to grasp, swing, and hit accurately
- Final evaluation:
  - Success rate: approximately 72% over 50 test episodes
  - Average reward: ~68.3
- Saved models:
  - `ppo_franka_golf.zip`
  - `franka_golf_final_model.pth`

---


## Installation

To install all required dependencies:

```bash
pip install -r requirements.txt

