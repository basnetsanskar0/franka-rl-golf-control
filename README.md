# Autonomous Robotic Arm Control via Reinforcement Learning 🏌️‍♂️🤖

This project trains a 7-DOF Franka Emika robotic arm to play a mini-golf game using deep reinforcement learning.

It was developed as part of the **ArenaX Labs Machine Learning Hiring Challenge** — where the objective is to teach a robot to:
- Grasp a golf club
- Align and swing with precision
- Hit a ball into a hole
- Avoid dropping the club or overshooting the hole

---

## 🧠 Approach Overview

This repo includes **two complete reinforcement learning pipelines**, each valuing different strengths:

### 1. `train_golf_bot.py` — Stable-Baselines3 PPO (High Reliability)
Uses Stable-Baselines3 to quickly train an agent with a custom neural network.

- ✅ Quick to prototype, easy to run
- 🧰 Leverages proven PPO implementation
- 📦 Produces a submission-ready `.zip` model
- 🎯 Good baseline for evaluation

### 2. `franka_golf_training.py` — Custom PyTorch PPO (High Flexibility)
Implements the full PPO algorithm from scratch, including:

- 🧠 Custom actor-critic network (learnable std dev)
- 🎯 Curriculum reward shaping: `grasp → align → swing → precision`
- 📊 Success tracking, GAE, manual updates
- 💾 Checkpoint saving, metrics logging, evaluation pipeline

This approach gives full transparency and control over learning behavior — suitable for fine-tuning or research-style development.

---

## ⚙️ Environment

- `FrankaIkGolfCourseEnv-v0` via [`sai-rl`](https://sai.arena-x.ai)
- 7D continuous action space:  
  `[Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]`
- 31D observation space:  
  Joint states + ball/club/hole info

---

## 🧪 Results

- ✅ Trained robot learns to grasp, swing, and hit accurately
- 📈 Final evaluation:  
  - **Success Rate**: ~72% over 50 test episodes  
  - **Average Reward**: ~+68.3
- 💾 Saved models:
  - `ppo_franka_golf.zip`
  - `franka_golf_final_model.pth`

---

## 🧰 How to Run

### Install Dependencies
```bash
pip install sai-rl gymnasium stable-baselines3 torch
