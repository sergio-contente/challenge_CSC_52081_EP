from student_client import create_student_env
import gym
import numpy as np

class SB3Env(gym.Env):
		def __init__(self):
				super(SB3Env, self).__init__()
				self.env = create_student_env(user_token='SERgio26735540')

		def step(self, action):
				obs, reward, done, info = self.env.step(action)
				return obs, reward, done, info

		def reset(self):
				return self.env.reset()

		def render(self, mode='human'):
				return self.env.render(mode)

		def close(self):
				self.env.close()
