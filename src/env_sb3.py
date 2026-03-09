import numpy as np
import gymnasium
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from student_client import create_student_gym_env
from student_client.student_gym_env_vectorized import create_student_gym_env_vectorized


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
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 2:
            obs = obs[-1]
        return obs

class VecSB3Env(VecEnv):
    """SB3 VecEnv backed by the student_client vectorized environment.

    Uses batch HTTP endpoints so N envs = 1 round-trip per step.
    Handles auto-reset internally (SB3 VecEnv contract).
    """

    def __init__(self, user_token: str, num_envs: int = 4):
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        act_space = spaces.Discrete(3)
        super().__init__(num_envs, obs_space, act_space)

        self.venv = create_student_gym_env_vectorized(
            user_token=user_token,
            num_envs=num_envs,
            auto_reset=False,
            return_all_states=False,
        )

        self.repair_counts = np.zeros(num_envs, dtype=int)
        self.steps_since_repair = np.zeros(num_envs, dtype=int)
        self.episode_steps = np.zeros(num_envs, dtype=int)

        self._actions = None
        self._obs_buf = np.zeros((num_envs, 9), dtype=np.float32)

    def reset(self):
        obs, _infos = self.venv.reset()
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        self._obs_buf[:] = obs
        self.repair_counts[:] = 0
        self.steps_since_repair[:] = 0
        self.episode_steps[:] = 0
        return self._obs_buf.copy()

    def step_async(self, actions):
        self._actions = np.asarray(actions, dtype=int)

    def step_wait(self):
        obs, rewards, terminateds, truncateds, infos = self.venv.step(
            self._actions, return_all_states=False
        )
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        rewards = np.asarray(rewards, dtype=np.float32)
        terminateds = np.asarray(terminateds, dtype=bool)
        truncateds = np.asarray(truncateds, dtype=bool)
        dones = terminateds | truncateds

        for i in range(self.num_envs):
            self.episode_steps[i] += 1
            if self._actions[i] == 1:
                self.repair_counts[i] += 1
                self.steps_since_repair[i] = 0
            else:
                self.steps_since_repair[i] += 1

            infos[i]["repair_count"] = int(self.repair_counts[i])
            infos[i]["episode_step"] = int(self.episode_steps[i])

        done_indices = np.where(dones)[0]
        if len(done_indices) > 0:
            for i in done_indices:
                infos[i]["terminal_observation"] = obs[i].copy()

            reset_obs, _reset_infos = self.venv.reset_specific_envs(done_indices.tolist())
            reset_obs = np.asarray(reset_obs, dtype=np.float32)
            for j, i in enumerate(done_indices):
                obs[i] = reset_obs[j]
                self.repair_counts[i] = 0
                self.steps_since_repair[i] = 0
                self.episode_steps[i] = 0

        self._obs_buf[:] = obs
        return self._obs_buf.copy(), rewards, dones, infos

    def close(self):
        self.venv.close()

    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise NotImplementedError

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def get_attr(self, attr_name, indices=None):
        raise AttributeError(attr_name)

    def set_attr(self, attr_name, value, indices=None):
        raise AttributeError(attr_name)

    def seed(self, seed=None):
        pass
