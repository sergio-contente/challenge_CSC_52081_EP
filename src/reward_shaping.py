import numpy as np


def shape_reward(raw_reward, action, obs, episode_step, terminated, truncated, info):
    shaped = raw_reward

    # Small survival bonus each step to encourage keeping engines running
    if not terminated and not truncated:
        shaped += 5.0

    # Penalize early selling (before step 10)
    if action == 2 and episode_step < 10:
        shaped -= 50.0

    # Bonus for selling at a good time (after running for a while)
    if action == 2 and episode_step >= 20:
        shaped += 20.0

    # Penalize engine failure (terminated without a sell action)
    if terminated and action != 2:
        shaped -= 100.0

    return shaped
