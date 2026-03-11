import sys
import os
import glob
import signal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import VecNormalize
from src.env_sb3 import VecSB3Env
from src.callbacks import EpisodeLoggerCallback, StepBudgetCallback

USER_TOKEN = "SERgio26735540"
NUM_ENVS = 4
TOTAL_TIMESTEPS = 20000
CHECKPOINT_DIR = "./checkpoints/ppo/"
CHECKPOINT_FREQ = 500


def find_latest_checkpoint(checkpoint_dir, prefix="ppo_aircraft"):
    patterns = [
        os.path.join(checkpoint_dir, f"{prefix}_ep*.zip"),
        os.path.join(checkpoint_dir, f"{prefix}_*_steps.zip"),
        os.path.join(checkpoint_dir, f"{prefix}_interrupted.zip"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def main():
    raw_env = VecSB3Env(user_token=USER_TOKEN, num_envs=NUM_ENVS)
    vecnorm_path = os.path.join(CHECKPOINT_DIR, "vecnormalize.pkl")

    checkpoint = find_latest_checkpoint(CHECKPOINT_DIR)

    if checkpoint and os.path.exists(vecnorm_path):
        print(f"Resuming from checkpoint: {checkpoint}")
        env = VecNormalize.load(vecnorm_path, raw_env)
        model = PPO.load(checkpoint, env=env, tensorboard_log="./logs/ppo/")
        remaining = max(0, TOTAL_TIMESTEPS - model.num_timesteps)
        print(f"Already trained {model.num_timesteps} steps, {remaining} remaining")
    else:
        env = VecNormalize(raw_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        if checkpoint:
            print(f"Resuming from checkpoint (no vecnorm stats): {checkpoint}")
            model = PPO.load(checkpoint, env=env, tensorboard_log="./logs/ppo/")
            remaining = max(0, TOTAL_TIMESTEPS - model.num_timesteps)
        else:
            print("Starting PPO training from scratch")
            remaining = TOTAL_TIMESTEPS
            model = PPO(
                policy="MlpPolicy",
                env=env,
                learning_rate=3e-4,
                n_steps=128,
                batch_size=64,
                n_epochs=4,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.05,
                vf_coef=0.5,
                max_grad_norm=0.5,
                policy_kwargs=dict(net_arch=[128, 128]),
                tensorboard_log="./logs/ppo/",
                verbose=1,
                seed=42,
            )

    # save on Ctrl+C
    def save_on_interrupt(signum, frame):
        print("\n[INTERRUPT] Saving model before exit...")
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        model.save(os.path.join(CHECKPOINT_DIR, "ppo_aircraft_interrupted"))
        model.save("models/ppo_aircraft")
        env.save(vecnorm_path)
        env.save("models/ppo_vecnormalize.pkl")
        print(f"[INTERRUPT] Saved model + vecnorm stats")
        env.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, save_on_interrupt)

    callbacks = CallbackList([
        EpisodeLoggerCallback(log_every_n_episodes=20),
        StepBudgetCallback(max_steps=TOTAL_TIMESTEPS),
        CheckpointCallback(save_freq=CHECKPOINT_FREQ, save_path=CHECKPOINT_DIR, name_prefix="ppo_aircraft"),
    ])

    if remaining > 0:
        model.learn(total_timesteps=remaining, callback=callbacks, reset_num_timesteps=False)
        model.save("models/ppo_aircraft")
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        env.save(vecnorm_path)
        env.save("models/ppo_vecnormalize.pkl")
        print(f"Model + vecnorm stats saved")
    else:
        print("Training already complete!")

    env.close()

if __name__ == "__main__":
    main()
