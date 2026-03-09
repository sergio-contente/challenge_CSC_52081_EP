import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from src.env_sb3 import VecSB3Env
from src.callbacks import EpisodeLoggerCallback, StepBudgetCallback

USER_TOKEN = "SERgio26735540"
NUM_ENVS = 4
TOTAL_TIMESTEPS = 5000

def main():
    env = VecSB3Env(user_token=USER_TOKEN, num_envs=NUM_ENVS)

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=500,
        batch_size=64,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
        target_update_interval=250,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[128, 128]),
        tensorboard_log="./logs/dqn/",
        verbose=1,
        seed=42,
    )

    callbacks = CallbackList([
        EpisodeLoggerCallback(log_every_n_episodes=20),
        StepBudgetCallback(max_steps=TOTAL_TIMESTEPS),
        CheckpointCallback(save_freq=1000, save_path="./checkpoints/dqn/", name_prefix="dqn_aircraft"),
    ])

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)
    model.save("models/dqn_aircraft")
    print(f"Model saved to models/dqn_aircraft")

    env.close()

if __name__ == "__main__":
    main()
