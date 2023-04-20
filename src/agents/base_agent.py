class BaseAgent:
    def __init__(self, config = None):
        self.config = config
        self.action_space = None
        self.observation_space = None
        
    def choose_action(self, state):
        raise NotImplementedError

    def learn(self, transition):
        raise NotImplementedError

    def save(self, filepath):
        raise NotImplementedError

    def load(self, filepath):
        raise NotImplementedError
    