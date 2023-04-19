import gym

class BaseEnvironment:
    def __init__(self, config):
        self.config = config
        self.env = gym.make(self.config["env_name"])
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()