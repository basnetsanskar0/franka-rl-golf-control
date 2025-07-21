#!/usr/bin/env python3
"""
Franka Golf RL Training Pipeline
Advanced manipulation task for ArenaX Labs ML Hiring Challenge
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, List
import gymnasium as gym
from sai_rl import SAIClient

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class FrankaGolfNetwork(nn.Module):
    """
    Actor-Critic network for Franka Golf manipulation task.
    Uses separate networks for policy (actor) and value estimation (critic).
    """
    
    def __init__(self, obs_dim: int = 31, action_dim: int = 7, hidden_dims: List[int] = [512, 512, 256]):
        super(FrankaGolfNetwork, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Shared feature extraction layers
        self.shared_layers = nn.ModuleList()
        prev_dim = obs_dim
        for hidden_dim in hidden_dims[:-1]:
            self.shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Final shared layer
        self.shared_final = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.LayerNorm(hidden_dims[-1]),
            nn.ReLU()
        )
        
        # Actor head (policy network)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dims[-1], 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()  # Actions are bounded [-1, 1]
        )
        
        # Actor log std (learnable parameter)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head (value network)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dims[-1], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.constant_(m.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning action mean, log std, and state value.
        """
        # Shared feature extraction
        x = obs
        for layer in self.shared_layers:
            x = layer(x)
        
        features = self.shared_final(x)
        
        # Actor output
        action_mean = self.actor_mean(features)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        
        # Critic output
        value = self.critic(features)
        
        return action_mean, action_log_std, value.squeeze(-1)
    
    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor = None):
        """
        Get action distribution and value for given observation.
        If action is provided, return log probability and entropy.
        """
        action_mean, action_log_std, value = self.forward(obs)
        action_std = torch.exp(action_log_std)
        
        probs = torch.distributions.Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value


class RewardShaper:
    """
    Advanced reward shaping for golf manipulation task.
    Provides curriculum learning and adaptive reward scaling.
    """
    
    def __init__(self):
        self.phase = "grasp"  # grasp -> align -> swing -> precision
        self.success_count = 0
        self.total_episodes = 0
        self.phase_thresholds = {
            "grasp": 0.6,    # 60% grasp success rate
            "align": 0.4,    # 40% alignment success rate
            "swing": 0.3,    # 30% swing success rate
        }
    
    def shape_reward(self, obs: np.ndarray, reward: float, info: dict) -> float:
        """
        Shape reward based on current learning phase and sub-task completion.
        """
        shaped_reward = reward
        
        # Extract relevant observations
        joint_pos = obs[:9]
        joint_vel = obs[9:18]
        ball_pos = obs[18:21]
        club_pos = obs[21:24]
        club_orient = obs[24:28]
        hole_pos = obs[28:31]
        
        # Calculate distances and alignments
        ee_club_dist = np.linalg.norm(club_pos - self._get_ee_pos_from_joints(joint_pos))
        ball_hole_dist = np.linalg.norm(ball_pos - hole_pos)
        
        # Phase-specific reward shaping
        if self.phase == "grasp":
            # Focus on grasping the club
            if ee_club_dist < 0.05:  # Close to club
                shaped_reward += 5.0
            if ee_club_dist < 0.02:  # Very close
                shaped_reward += 10.0
                
        elif self.phase == "align":
            # Focus on alignment and positioning
            if ee_club_dist < 0.02:  # Has grasped club
                shaped_reward += 5.0
                # Reward ball-hole alignment
                ball_hole_alignment = self._calculate_alignment(ball_pos, hole_pos, club_pos)
                shaped_reward += ball_hole_alignment * 3.0
                
        elif self.phase == "swing":
            # Focus on making contact and ball movement
            ball_velocity = self._estimate_ball_velocity(obs)
            if np.linalg.norm(ball_velocity) > 0.01:  # Ball is moving
                shaped_reward += 8.0
                # Reward movement toward hole
                if ball_hole_dist < np.linalg.norm(self.prev_ball_pos - hole_pos) if hasattr(self, 'prev_ball_pos') else True:
                    shaped_reward += 5.0
        
        # Store previous ball position for velocity estimation
        self.prev_ball_pos = ball_pos.copy()
        
        # Smooth movement bonus
        joint_vel_penalty = -np.sum(np.square(joint_vel)) * 0.001
        shaped_reward += joint_vel_penalty
        
        # Success bonus
        if info.get('success', False):
            shaped_reward += 100.0
            self.success_count += 1
        
        return shaped_reward
    
    def _get_ee_pos_from_joints(self, joint_pos: np.ndarray) -> np.ndarray:
        """Approximate end-effector position from joint positions."""
        # This is a simplified approximation - in practice you'd use forward kinematics
        # For now, we'll use the club position as a proxy since the environment provides it
        return np.array([0.5, 0.0, 0.8])  # Default EE position
    
    def _calculate_alignment(self, ball_pos: np.ndarray, hole_pos: np.ndarray, club_pos: np.ndarray) -> float:
        """Calculate how well the club is aligned for hitting ball toward hole."""
        ball_to_hole = hole_pos - ball_pos
        club_to_ball = ball_pos - club_pos
        
        if np.linalg.norm(ball_to_hole) > 0 and np.linalg.norm(club_to_ball) > 0:
            alignment = np.dot(club_to_ball, ball_to_hole) / (
                np.linalg.norm(club_to_ball) * np.linalg.norm(ball_to_hole)
            )
            return max(0, alignment)  # Only positive alignment
        return 0.0
    
    def _estimate_ball_velocity(self, obs: np.ndarray) -> np.ndarray:
        """Estimate ball velocity from position changes."""
        if hasattr(self, 'prev_ball_pos'):
            ball_pos = obs[18:21]
            return ball_pos - self.prev_ball_pos
        return np.zeros(3)
    
    def update_phase(self, episode_rewards: List[float]):
        """Update learning phase based on performance."""
        self.total_episodes += 1
        if self.total_episodes % 100 == 0:  # Check every 100 episodes
            success_rate = self.success_count / 100
            
            if self.phase == "grasp" and success_rate > self.phase_thresholds["grasp"]:
                self.phase = "align"
                print(f"Advanced to alignment phase (success rate: {success_rate:.2f})")
            elif self.phase == "align" and success_rate > self.phase_thresholds["align"]:
                self.phase = "swing"
                print(f"Advanced to swing phase (success rate: {success_rate:.2f})")
            elif self.phase == "swing" and success_rate > self.phase_thresholds["swing"]:
                self.phase = "precision"
                print(f"Advanced to precision phase (success rate: {success_rate:.2f})")
            
            self.success_count = 0


class PPOTrainer:
    """
    Proximal Policy Optimization (PPO) trainer optimized for manipulation tasks.
    """
    
    def __init__(self, 
                 network: FrankaGolfNetwork,
                 learning_rate: float = 3e-4,
                 clip_epsilon: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 gae_lambda: float = 0.95,
                 gamma: float = 0.99):
        
        self.network = network
        self.optimizer = optim.Adam(network.parameters(), lr=learning_rate, eps=1e-5)
        
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)
        
    def compute_gae(self, rewards: List[float], values: List[float], dones: List[bool]) -> Tuple[List[float], List[float]]:
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                nextnonterminal = 1.0 - dones[t]
                nextvalue = 0
            else:
                nextnonterminal = 1.0 - dones[t+1]
                nextvalue = values[t+1]
            
            delta = rewards[t] + self.gamma * nextvalue * nextnonterminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * nextnonterminal * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns
    
    def update(self, batch_obs, batch_actions, batch_returns, batch_advantages, batch_logprobs):
        """Perform PPO update."""
        # Convert to tensors
        obs = torch.FloatTensor(np.array(batch_obs)).to(self.device)
        actions = torch.FloatTensor(np.array(batch_actions)).to(self.device)
        returns = torch.FloatTensor(batch_returns).to(self.device)
        advantages = torch.FloatTensor(batch_advantages).to(self.device)
        old_logprobs = torch.FloatTensor(batch_logprobs).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(4):  # Multiple epochs
            # Get current policy outputs
            _, newlogprobs, entropy, values = self.network.get_action_and_value(obs, actions)
            
            # Calculate ratio
            ratio = torch.exp(newlogprobs - old_logprobs)
            
            # Calculate surrogates
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            
            # Calculate losses
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, returns)
            entropy_loss = entropy.mean()
            
            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item()
        }


def train_franka_golf(num_episodes: int = 5000, 
                     batch_size: int = 2048,
                     save_interval: int = 500):
    """Main training loop for Franka Golf."""
    
    # Initialize environment
    try:
        sai = SAIClient("FrankaIkGolfCourseEnv-v0")
        env = sai.make_env()
    except:
        # Fallback to gym
        import sai_mujoco
        env = gym.make("FrankaIkGolfCourseEnv-v0")
    
    # Initialize components
    network = FrankaGolfNetwork()
    trainer = PPOTrainer(network)
    reward_shaper = RewardShaper()
    
    # Training metrics
    episode_rewards = []
    success_rates = []
    recent_rewards = deque(maxlen=100)
    
    # Training variables
    obs_buffer = []
    action_buffer = []
    reward_buffer = []
    value_buffer = []
    logprob_buffer = []
    done_buffer = []
    
    print("Starting Franka Golf training...")
    print(f"Device: {trainer.device}")
    
    obs, _ = env.reset()
    episode_reward = 0
    episode_count = 0
    
    for step in range(num_episodes * 650):  # Max 650 steps per episode
        # Get action from policy
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(trainer.device)
        with torch.no_grad():
            action, logprob, entropy, value = network.get_action_and_value(obs_tensor)
        
        action_np = action.cpu().numpy().squeeze()
        
        # Take step in environment
        next_obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated
        
        # Shape reward
        shaped_reward = reward_shaper.shape_reward(obs, reward, info)
        
        # Store transition
        obs_buffer.append(obs)
        action_buffer.append(action_np)
        reward_buffer.append(shaped_reward)
        value_buffer.append(value.item())
        logprob_buffer.append(logprob.item())
        done_buffer.append(done)
        
        episode_reward += reward
        obs = next_obs
        
        if done:
            episode_count += 1
            recent_rewards.append(episode_reward)
            
            if episode_count % 100 == 0:
                avg_reward = np.mean(recent_rewards)
                success_rate = sum(1 for r in recent_rewards if r > 50) / len(recent_rewards)  # Rough success threshold
                success_rates.append(success_rate)
                
                print(f"Episode {episode_count}: Avg Reward: {avg_reward:.2f}, Success Rate: {success_rate:.2f}")
                
                # Update curriculum phase
                reward_shaper.update_phase(list(recent_rewards))
            
            # Reset environment
            obs, _ = env.reset()
            episode_reward = 0
        
        # Update policy when buffer is full
        if len(obs_buffer) >= batch_size:
            # Compute advantages and returns
            advantages, returns = trainer.compute_gae(
                reward_buffer, value_buffer, done_buffer
            )
            
            # Perform update
            losses = trainer.update(
                obs_buffer, action_buffer, returns, advantages, logprob_buffer
            )
            
            # Clear buffers
            obs_buffer.clear()
            action_buffer.clear()
            reward_buffer.clear()
            value_buffer.clear()
            logprob_buffer.clear()
            done_buffer.clear()
            
            print(f"Policy Loss: {losses['policy_loss']:.4f}, "
                  f"Value Loss: {losses['value_loss']:.4f}")
        
        # Save model periodically
        if episode_count > 0 and episode_count % save_interval == 0:
            torch.save({
                'network_state_dict': network.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'episode': episode_count,
                'avg_reward': np.mean(recent_rewards) if recent_rewards else 0
            }, f'franka_golf_model_episode_{episode_count}.pth')
            print(f"Model saved at episode {episode_count}")
    
    # Final save
    torch.save({
        'network_state_dict': network.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'episode': episode_count,
        'final_avg_reward': np.mean(recent_rewards) if recent_rewards else 0,
        'success_rates': success_rates
    }, 'franka_golf_final_model.pth')
    
    env.close()
    return network, episode_rewards, success_rates


def evaluate_model(model_path: str, num_episodes: int = 100):
    """Evaluate trained model."""
    # Load environment
    try:
        sai = SAIClient("FrankaIkGolfCourseEnv-v0")
        env = sai.make_env(render_mode="human")
    except:
        import sai_mujoco
        env = gym.make("FrankaIkGolfCourseEnv-v0", render_mode="human")
    
    # Load model
    network = FrankaGolfNetwork()
    checkpoint = torch.load(model_path, map_location='cpu')
    network.load_state_dict(checkpoint['network_state_dict'])
    network.eval()
    
    episode_rewards = []
    success_count = 0
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        
        for step in range(650):  # Max steps per episode
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _ = network.get_action_and_value(obs_tensor)
            
            action_np = action.cpu().numpy().squeeze()
            obs, reward, terminated, truncated, info = env.step(action_np)
            episode_reward += reward
            
            if terminated or truncated:
                if info.get('success', False):
                    success_count += 1
                break
        
        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward: {episode_reward:.2f}")
    
    avg_reward = np.mean(episode_rewards)
    success_rate = success_count / num_episodes
    
    print(f"\nEvaluation Results:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Success Rate: {success_rate:.2f}")
    print(f"Successful Episodes: {success_count}/{num_episodes}")
    
    env.close()
    return episode_rewards, success_rate


if __name__ == "__main__":
    # Train the model
    print("Training Franka Golf model...")
    network, rewards, success_rates = train_franka_golf(
        num_episodes=3000,
        batch_size=2048,
        save_interval=500
    )
    
    # Plot training progress
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(success_rates)
    plt.title('Success Rate')
    plt.xlabel('Episode (x100)')
    plt.ylabel('Success Rate')
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.show()
    
    print("Training completed! Model saved as 'franka_golf_final_model.pth'")
    
    # Evaluate the final model
    print("\nEvaluating final model...")
    eval_rewards, final_success_rate = evaluate_model(
        'franka_golf_final_model.pth', 
        num_episodes=50
    )
    
    print(f"Final evaluation success rate: {final_success_rate:.2f}")