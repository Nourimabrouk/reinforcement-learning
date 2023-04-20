from .base_agent import BaseAgent
import gymnasium as gym

class RandomAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(2)
        
    def choose_action(self, state):
        return int(self.action_space.sample())

    def learn(self, transition):
        pass

    def save(self, filepath):
        pass

    def load(self, filepath):
        pass