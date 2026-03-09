import sys
import os
import glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from src.env_sb3 import VecSB3Env
from src.callbacks import EpisodeLoggerCallback, StepBudgetCallback

USER_TOKEN = "SERgio26735540"
NUM_ENVS = 4
TOTAL_TIMESTEPS = 20000
CHECKPOINT_DIR = "./checkpoints/dqn/"
CHECKPOINT_FREQ = 500


def find_latest_checkpoint(checkpoint_dir, prefix="dqn_aircraft"):
    pattern = os.path.join(checkpoint_dir, f"{prefix}_*_steps.zip")
    files = glob.glob(pattern)
    if not files:
        return None
    latest = max(files, key=os.path.getmtime)
    return latest


def main():
    env = VecSB3Env(user_token=USER_TOKEN, num_envs=NUM_ENVS)

    checkpoint = find_latest_checkpoint(CHECKPOINT_DIR)

    if checkpoint:
        print(f"Resuming from checkpoint: {checkpoint}")
        model = DQN.load(checkpoint, env=env, tensorboard_log="./logs/dqn/")
        remaining = max(0, TOTAL_TIMESTEPS - model.num_timesteps)
        print(f"Already trained {model.num_timesteps} steps, {remaining} remaining")
    else:
        print("Starting training from scratch")
        remaining = TOTAL_TIMESTEPS
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
        CheckpointCallback(save_freq=CHECKPOINT_FREQ, save_path=CHECKPOINT_DIR, name_prefix="dqn_aircraft"),
    ])

    if remaining > 0:
        model.learn(total_timesteps=remaining, callback=callbacks, reset_num_timesteps=False)
        model.save("models/dqn_aircraft")
        print(f"Model saved to models/dqn_aircraft")
    else:
        print("Training already complete!")

    env.close()

if __name__ == "__main__":
    main()
