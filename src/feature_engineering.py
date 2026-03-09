import numpy as np

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
        self.window_buf = np.zeros((num_envs, WINDOW_SIZE, 7), dtype=np.float32)
        self.window_counts = np.zeros(num_envs, dtype=int)

    def reset(self, env_indices=None):
        if env_indices is None:
            self.window_buf[:] = 0
            self.window_counts[:] = 0
        else:
            for i in env_indices:
                self.window_buf[i] = 0
                self.window_counts[i] = 0

    def _push(self, env_idx, sensors):
        """Push sensor reading into ring buffer for env_idx."""
        c = self.window_counts[env_idx]
        pos = c % WINDOW_SIZE
        self.window_buf[env_idx, pos] = sensors
        self.window_counts[env_idx] = c + 1

    def _get_window(self, env_idx):
        """Return stored observations in chronological order."""
        c = self.window_counts[env_idx]
        n = min(c, WINDOW_SIZE)
        if n == 0:
            return np.zeros((0, 7), dtype=np.float32)
        if c <= WINDOW_SIZE:
            return self.window_buf[env_idx, :n].copy()
        # ring buffer wrapped: reorder
        start = c % WINDOW_SIZE
        indices = [(start + j) % WINDOW_SIZE for j in range(n)]
        return self.window_buf[env_idx, indices]

    def transform(self, raw_obs, episode_steps, repair_counts, steps_since_repair):
        """Transform (num_envs, 9) raw obs into (num_envs, NUM_FEATURES)."""
        out = np.zeros((self.num_envs, NUM_FEATURES), dtype=np.float32)

        for i in range(self.num_envs):
            obs = raw_obs[i]
            sensors = obs[SENSOR_INDICES]
            self._push(i, sensors)

            # [0:9] raw obs
            out[i, :NUM_RAW] = obs

            w = self._get_window(i)
            n = len(w)

            # [9:16] delta: current - previous
            if n >= 2:
                out[i, 9:16] = w[-1] - w[-2]

            # [16:23] slope: simple (last - first) / (n - 1)
            if n >= 3:
                out[i, 16:23] = (w[-1] - w[0]) / (n - 1)

            # [23:30] rolling mean
            out[i, 23:30] = w.mean(axis=0) if n >= 1 else sensors

            # [30:33] meta features (raw)
            out[i, 30] = float(episode_steps[i])
            out[i, 31] = float(repair_counts[i])
            out[i, 32] = float(steps_since_repair[i])

        return out
