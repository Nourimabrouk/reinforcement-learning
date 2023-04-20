import gym
from gym import spaces
import numpy as np


class GridWorld(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 30}

    def __init__(self):
        super().__init__()

        self.observation_space = gym.spaces.Tuple((gym.spaces.Discrete(5), gym.spaces.Discrete(5)))
        self.action_space = gym.spaces.Discrete(4)

        self.agent_position = (0, 0)
        self.done = False
        self.truncated = False
        self.info = {}

    def reset(self):
        self.agent_position = (0, 0)
        self.done = False
        self.truncated = False
        self.info = {}
        return self.agent_position

    def step(self, action):
        if self.done:
            raise ValueError("Cannot step in a terminal state")

        x, y = self.agent_position
        reward = -1
        if action == 0:  # Move up
            y = max(0, y - 1)
        elif action == 1:  # Move down
            y = min(4, y + 1)
        elif action == 2:  # Move left
            x = max(0, x - 1)
        elif action == 3:  # Move right
            x = min(4, x + 1)

        self.agent_position = (x, y)

        if self.agent_position == (4, 4):  # Goal position
            self.done = True
            reward = 100

        observation = self.agent_position
        info = self.info
        terminated = self.done
        truncated = self.truncated

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        grid = [['.' for _ in range(5)] for _ in range(5)]
        grid[4][4] = 'G'
        x, y = self.agent_position
        grid[y][x] = 'A'

        for row in grid:
            print(' '.join(row))
        print()