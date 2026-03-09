import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3.common.env_checker import check_env
from src.env_sb3 import SB3Env

env = SB3Env(user_token="SERgio26735540")
check_env(env)
print("check_env passed!")

obs, info = env.reset()
print(f"reset obs shape: {obs.shape}, dtype: {obs.dtype}")

for i in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"step {i}: action={action}, reward={reward:.2f}, obs shape={obs.shape}, done={terminated}")
    if terminated or truncated:
        break

env.close()
