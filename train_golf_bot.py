# train_golf_bot.py

from sai_rl import SAIClient
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
import torch as th
import torch.nn as nn
import gymnasium as gym
import numpy as np

# --- Optional: Custom network class ---
class CustomGolfNet(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.net(observations)

# --- Connect to competition ---
sai = SAIClient(comp_id="franka-ml-hiring")  # Correct comp ID
env = FlattenObservation(sai.make_env())     # Flatten obs for neural net

# --- Use a custom policy that includes our custom net ---
policy_kwargs = dict(
    features_extractor_class=CustomGolfNet,
    features_extractor_kwargs=dict(features_dim=128)
)

# --- Optional: Parallel envs (for speed) ---
# env = make_vec_env(lambda: env, n_envs=4)

# --- Initialize PPO agent ---
model = PPO(
    policy="MlpPolicy",
    env=env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    n_steps=2048,
    batch_size=64,
    gae_lambda=0.95,
    gamma=0.99,
    ent_coef=0.01,
    learning_rate=3e-4,
    vf_coef=0.5,
    max_grad_norm=0.5,
)

# --- Optional: Save checkpoints every X timesteps ---
checkpoint_callback = CheckpointCallback(
    save_freq=50_000, save_path="./checkpoints", name_prefix="franka_model"
)

# --- Train the agent ---
model.learn(total_timesteps=500_000, callback=checkpoint_callback)
model.save("ppo_franka_golf")

# --- Evaluation (1 episode) ---
obs, _ = env.reset()
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()
