def shape_reward(raw_reward, action, obs, episode_step, terminated, truncated, info):
    shaped = raw_reward

    steps_since_repair = obs[32]  # raw value (not normalized)

    # --- Strong survival bonus: make staying alive clearly valuable ---
    if not terminated and not truncated:
        shaped += 50.0

    # --- Sell decisions ---
    if action == 2:
        if episode_step < 10:
            shaped -= 500.0  # crushing penalty for selling too early
        else:
            shaped += episode_step * 5.0  # bonus for selling later

    # --- Engine failure: moderate penalty (don't scare agent from exploring) ---
    if terminated and action != 2:
        shaped -= 100.0

    # --- Repair decisions based on time since last repair ---
    if action == 1:
        if episode_step <= 2:
            shaped -= 20.0  # repairing a fresh engine
        elif steps_since_repair >= 15:
            shaped += 10.0  # been a while, likely needed
        else:
            shaped -= 15.0  # repaired too recently

    return shaped
