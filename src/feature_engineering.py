import numpy as np
from collections import deque

# Raw observation indices
# 0: HPC_Tout, 1: HP_Nmech, 2: HPC_Tin, 3: LPT_Tin,
# 4: Fuel_flow, 5: HPC_Pout_st, 6: LP_Nmech, 7: phase_type, 8: DTAMB

NUM_RAW = 9
SENSOR_INDICES = list(range(7))  # indices 0-6 are sensor readings
WINDOW_SIZE = 5

# Feature vector layout:
# [0:9]   raw obs
# [9:16]  delta (current - previous) for 7 sensors
# [16:23] slope (linear regression over window) for 7 sensors
# [23:30] rolling mean over window for 7 sensors
# [30]    episode_step (normalized)
# [31]    repair_count (normalized)
# [32]    steps_since_repair (normalized)
NUM_FEATURES = NUM_RAW + 7 * 3 + 3  # 9 + 21 + 3 = 33


class FeatureExtractor:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.raw_buf = np.zeros((num_envs, NUM_RAW), dtype=np.float32)
        self.delta_buf = np.zeros((num_envs, 7), dtype=np.float32)
        self.slope_buf = np.zeros((num_envs, 7), dtype=np.float32)
        self.mean_buf = np.zeros((num_envs, 7), dtype=np.float32)
        self.window_buf = np.zeros((num_envs, WINDOW_SIZE, 7), dtype=np.float32)
        self.window_counts = np.zeros(num_envs, dtype=int)
    