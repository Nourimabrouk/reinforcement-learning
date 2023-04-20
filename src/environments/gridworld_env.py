import gym
from gym import spaces
import numpy as np

class GridWorld(gym.Env):
    def __init__(self, config=None):
        super().__init__()

        self.observation_space = spaces.Box(low=0, high=1, shape=(4, 4), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)  

        self.grid = np.zeros((4, 4))
        self.state = (0, 0)
        self.goal = (3, 3)

    def reset(self):
        self.state = (0, 0)
        return self.state

    def step(self, action):
        x, y = self.state

        if action == 0:
            x = np.clip(x - 1, 0, 3)
        elif action == 1:
            x = np.clip(x + 1, 0, 3)
        elif action == 2:
            y = np.clip(y - 1, 0, 3)
        elif action == 3:
            y = np.clip(y + 1, 0, 3)

        self.state = (x, y)

        done = self.state == self.goal
        reward = 1 if done else -1

        return self.state, reward, done, {}

    def render(self, mode='human'):
        grid_repr = np.zeros_like(self.grid, dtype=str)
        grid_repr[self.grid == 0] = '.'
        grid_repr[self.state] = 'A'
        grid_repr[self.goal] = 'G'
        print(grid_repr)