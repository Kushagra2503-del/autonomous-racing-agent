import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os

# We use the "continuous=False" setting to simplify the actions (Left, Right, Gas, Brake)
env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=False)
env = DummyVecEnv([lambda: env])


# We use PPO (Proximal Policy Optimization) with a CNN Policy because the input is pixels (images).
model = PPO(
    "CnnPolicy", 
    env, 
    verbose=1, 
    learning_rate=0.0003,
)


print("Starting training...")
# For a demo, we train for 100k steps. 
# In a real scenario, this would be 200k-500k steps.
model.learn(total_timesteps=100000)


model.save("pro_racer_final")
print("Training finished and model saved!")
