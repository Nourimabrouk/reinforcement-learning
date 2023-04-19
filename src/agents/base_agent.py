class BaseAgent:
    def __init__(self, config):
        self.config = config

    def choose_action(self, state):
        raise NotImplementedError

    def learn(self, transition):
        raise NotImplementedError

    def save(self, filepath):
        raise NotImplementedError

    def load(self, filepath):
        raise NotImplementedError