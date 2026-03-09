import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeLoggerCallback(BaseCallback):
    def __init__(self, log_every_n_episodes: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.log_every = log_every_n_episodes
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_repairs = []
        self.failure_count = 0
        self.sell_count = 0
        self.total_episodes = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, done in enumerate(dones):
            if not done:
                continue

            info = infos[i]
            ep_reward = info.get("total_reward", 0.0)
            ep_length = info.get("episode_step", 0)
            ep_repairs = info.get("repair_count", 0)

            self.episode_rewards.append(ep_reward)
            self.episode_lengths.append(ep_length)
            self.episode_repairs.append(ep_repairs)
            self.total_episodes += 1

            if ep_reward < 0:
                self.failure_count += 1
            else:
                self.sell_count += 1

            # log every episode to TensorBoard
            self.logger.record("episode/reward", ep_reward)
            self.logger.record("episode/length", ep_length)
            self.logger.record("episode/repairs", ep_repairs)
            self.logger.record("episode/total_episodes", self.total_episodes)
            self.logger.record("episode/failure_rate", self.failure_count / self.total_episodes)

            # rolling averages (last N episodes)
            window = min(50, len(self.episode_rewards))
            self.logger.record("episode/avg_reward_50", np.mean(self.episode_rewards[-window:]))
            self.logger.record("episode/avg_length_50", np.mean(self.episode_lengths[-window:]))

            # print summary every N episodes
            if self.total_episodes % self.log_every == 0:
                recent = min(self.log_every, len(self.episode_rewards))
                avg_r = np.mean(self.episode_rewards[-recent:])
                avg_l = np.mean(self.episode_lengths[-recent:])
                avg_rep = np.mean(self.episode_repairs[-recent:])
                fail_rate = self.failure_count / self.total_episodes
                print(
                    f"[EP {self.total_episodes}] "
                    f"avg_reward={avg_r:.1f}  avg_len={avg_l:.1f}  "
                    f"avg_repairs={avg_rep:.1f}  fail_rate={fail_rate:.0%}  "
                    f"timesteps={self.num_timesteps}"
                )

        return True


class StepBudgetCallback(BaseCallback):
    """Stops training when the API step budget is exhausted."""

    def __init__(self, max_steps: int = 40000, verbose: int = 0):
        super().__init__(verbose)
        self.max_steps = max_steps

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.max_steps:
            print(f"[BUDGET] Step budget of {self.max_steps} reached at timestep {self.num_timesteps}. Stopping.")
            return False
        return True
