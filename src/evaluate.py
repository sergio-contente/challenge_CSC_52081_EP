import sys
import os
import glob
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import DQN, PPO
from student_client import get_leaderboard_score
from src.env_sb3 import VecSB3Env

USER_TOKEN = "SERgio26735540"
NUM_ENVS = 4
NUM_EVAL_EPISODES = 100

ALGO_CONFIG = {
    "dqn": {"cls": DQN, "model_path": "models/dqn_aircraft", "checkpoint_dir": "./checkpoints/dqn/", "prefix": "dqn_aircraft"},
    "ppo": {"cls": PPO, "model_path": "models/ppo_aircraft", "checkpoint_dir": "./checkpoints/ppo/", "prefix": "ppo_aircraft"},
}


def find_latest_checkpoint(checkpoint_dir, prefix):
    patterns = [
        os.path.join(checkpoint_dir, f"{prefix}_ep*.zip"),
        os.path.join(checkpoint_dir, f"{prefix}_*_steps.zip"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def load_model(env, algo="dqn"):
    """Load model: try saved model first, then latest checkpoint."""
    cfg = ALGO_CONFIG[algo]
    model_path = cfg["model_path"]
    cls = cfg["cls"]

    if os.path.exists(model_path + ".zip"):
        print(f"Loading {algo.upper()} model from {model_path}")
        return cls.load(model_path, env=env)

    checkpoint = find_latest_checkpoint(cfg["checkpoint_dir"], cfg["prefix"])
    if checkpoint:
        print(f"Loading {algo.upper()} checkpoint from {checkpoint}")
        return cls.load(checkpoint, env=env)

    print(f"No {algo.upper()} model or checkpoint found!")
    sys.exit(1)


def evaluate(algo="dqn"):
    env = VecSB3Env(user_token=USER_TOKEN, num_envs=NUM_ENVS)
    model = load_model(env, algo=algo)

    print(f"Evaluating for {NUM_EVAL_EPISODES} episodes (deterministic policy)...")

    episode_rewards = []
    episode_lengths = []
    episode_repairs = []
    failures = 0

    obs = env.reset()
    episodes_done = 0

    while episodes_done < NUM_EVAL_EPISODES:
        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(actions)

        for i, done in enumerate(dones):
            if not done:
                continue

            episodes_done += 1
            raw_reward = infos[i].get("raw_reward", 0.0)
            ep_length = infos[i].get("episode_step", 0)
            ep_repairs = infos[i].get("repair_count", 0)

            episode_rewards.append(raw_reward)
            episode_lengths.append(ep_length)
            episode_repairs.append(ep_repairs)

            if raw_reward < 0:
                failures += 1

            if episodes_done % 10 == 0:
                avg_r = np.mean(episode_rewards[-10:])
                print(f"  [{episodes_done}/{NUM_EVAL_EPISODES}] last_10_avg={avg_r:.1f}")

            if episodes_done >= NUM_EVAL_EPISODES:
                break

    env.close()

    # print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Episodes:        {len(episode_rewards)}")
    print(f"Avg reward:      {np.mean(episode_rewards):.2f}")
    print(f"Std reward:      {np.std(episode_rewards):.2f}")
    print(f"Min reward:      {np.min(episode_rewards):.2f}")
    print(f"Max reward:      {np.max(episode_rewards):.2f}")
    print(f"Avg length:      {np.mean(episode_lengths):.1f}")
    print(f"Avg repairs:     {np.mean(episode_repairs):.1f}")
    print(f"Failure rate:    {failures / len(episode_rewards):.1%}")
    print(f"\nBaseline 1:      3951.44 (5% failure)")
    print(f"Baseline 2:      3111.07 (47% failure)")
    print("=" * 50)

    try:
        score = get_leaderboard_score(
            user_token=USER_TOKEN,
            server_url="http://rlchallenge.orailix.com",
            limit=100,
            return_dataframe=False,
        )
        print(f"Leaderboard: {score}")
    except Exception as e:
        print(f"Could not fetch leaderboard: {e}")


if __name__ == "__main__":
    algo = sys.argv[1] if len(sys.argv) > 1 else "dqn"
    if algo not in ALGO_CONFIG:
        print(f"Unknown algo '{algo}'. Choose from: {list(ALGO_CONFIG.keys())}")
        sys.exit(1)
    evaluate(algo=algo)
