import gym
import numpy as np


class Env():
    """
    """
    def __init__(self, envName):
        self.env = gym.make(envName)
        self.reset()
        self.observation_spacelow = self.env.observation_space.low[2]
        self.observation_spacehigh = self.env.observation_space.high[2]
        self.actionspace = self.env.action_space.n
    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()

    def reset(self):
        return self.env.reset()


