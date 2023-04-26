from .base_agent import BaseAgent
import gymnasium as gym
import numpy as np


class RandomAgent(BaseAgent):
    def __init__(self, env=None, config=None):
        super().__init__()
        self.action_space = gym.spaces.Discrete(2)
        self.env = env
        
    def choose_action(self, state):
        return int(self.action_space.sample())

    def learn(self, transition):
        pass

    def save(self, filepath):
        pass

    def load(self, filepath):
        pass
    
