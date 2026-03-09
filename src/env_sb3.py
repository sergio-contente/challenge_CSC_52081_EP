import numpy as np
import gymnasium
from gymnasium import spaces
from student_client import create_student_gym_env


class SB3Env(gymnasium.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, user_token: str):
        super().__init__()
        self.student_env = create_student_gym_env(
            user_token=user_token,
            auto_reset=False,
        )
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )

        self.repair_count = 0
        self.steps_since_repair = 0
        self.episode_step = 0
        self.prev_obs = None

    def reset(self, seed=None, options=None):
        obs, info = self.student_env.reset()
        obs = self._to_single_obs(obs)

        self.repair_count = 0
        self.steps_since_repair = 0
        self.episode_step = 0
        self.prev_obs = None
        
        return obs, info

    def step(self, action):
        action = int(action)
        result = self.student_env.step(action=action, step_size=10, return_all_states=False)

        obs, reward, terminated, truncated, info = result
        obs = self._to_single_obs(obs)

        self.episode_step += 1
        if action == 1:
            self.repair_count += 1
            self.steps_since_repair = 0
        else:
            self.steps_since_repair += 1
        self.prev_obs = obs.copy()

        info["repair_count"] = self.repair_count
        info["episode_step"] = self.episode_step

        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        self.student_env.close()

    @staticmethod
    def _to_single_obs(obs):
        #Guarantee a flat (9,) float32 array
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 2:
            obs = obs[-1]
        return obs
