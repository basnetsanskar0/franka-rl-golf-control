# Autonomous Robotic Arm Control via Reinforcement Learning

🏌️ Train a 7-DOF Franka Emika robotic arm to play mini-golf using deep reinforcement learning.

## 📦 Environment
- FrankaIkGolfCourseEnv-v0 (via `sai-rl`)
- Continuous 7D action space: [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]
- 31D observation space

## 🧠 Algorithm
- PPO (Proximal Policy Optimization)
- Actor-Critic neural network (custom PyTorch model)
- Reward shaping with curriculum learning

## 🧠 Key Challenges Tackled
- Sparse reward learning with GAE
- Curriculum phase-based reward shaping
- End-effector alignment and tool use coordination

## 🚀 How to Run

```bash
python franka_golf_training.py
