import numpy as np

# Feature vector layout (33 dims):
# [0:9]   raw obs (7 sensors + phase_type + DTAMB)
# [9:16]  delta (current - previous) for 7 sensors
# [16:23] slope (trend over window) for 7 sensors
# [23:30] rolling mean for 7 sensors
# [30]    episode_step
# [31]    repair_count
# [32]    steps_since_repair

# Sensor indices within raw obs [0:7]
HPC_TOUT = 0
HP_NMECH = 1
HPC_TIN = 2
LPT_TIN = 3
FUEL_FLOW = 4
HPC_POUT_ST = 5
LP_NMECH = 6


def shape_reward(raw_reward, action, obs, episode_step, terminated, truncated, info):
    shaped = raw_reward

    deltas = obs[9:16]
    slopes = obs[16:23]

    # --- Survival bonus ---
    if not terminated and not truncated:
        shaped += 5.0

    # --- Sell timing ---
    if action == 2:
        if episode_step < 10:
            shaped -= 50.0
        elif episode_step >= 20:
            shaped += 20.0

    # --- Engine failure penalty ---
    if terminated and action != 2:
        shaped -= 100.0

    # --- Smart repair: reward when sensors show degradation ---
    if action == 1 and episode_step > 2:
        # magnitude of sensor drift (deltas)
        delta_magnitude = np.abs(deltas).mean()
        # magnitude of degradation trend (slopes)
        slope_magnitude = np.abs(slopes).mean()

        if delta_magnitude > 0.3 or slope_magnitude > 0.2:
            shaped += 15.0  # good repair timing
        else:
            shaped -= 5.0  # unnecessary repair, wasting money

    # --- Penalize repair on fresh engine ---
    if action == 1 and episode_step <= 2:
        shaped -= 15.0

    # --- Penalize doing nothing when degradation is high ---
    if action == 0 and episode_step > 5:
        slope_magnitude = np.abs(slopes).mean()
        if slope_magnitude > 0.5:
            shaped -= 10.0  # should be repairing or selling

    return shaped
