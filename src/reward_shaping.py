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

    # --- Survival bonus (small, just to encourage not crashing) ---
    if not terminated and not truncated:
        shaped += 1.0

    # --- Sell bonus: reward selling after enough steps ---
    if action == 2:
        if episode_step < 5:
            shaped -= 50.0  # selling too early wastes potential
        elif episode_step >= 15:
            shaped += 30.0  # good sell timing

    # --- Engine failure penalty ---
    if terminated and action != 2:
        shaped -= 200.0

    # --- Repair: only reward when degradation is clearly high ---
    if action == 1:
        if episode_step <= 2:
            shaped -= 20.0  # repairing a fresh engine
        else:
            slope_magnitude = np.abs(slopes).mean()
            delta_magnitude = np.abs(deltas).mean()
            if slope_magnitude > 0.5 or delta_magnitude > 0.5:
                shaped += 5.0  # justified repair
            else:
                shaped -= 15.0  # unnecessary repair, heavy penalty

    # --- Penalize doing nothing when degradation is critical ---
    if action == 0 and episode_step > 10:
        slope_magnitude = np.abs(slopes).mean()
        if slope_magnitude > 1.0:
            shaped -= 5.0  # should be repairing or selling

    return shaped
