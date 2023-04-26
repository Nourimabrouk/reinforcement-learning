class BaseAgent:
    def __init__(self, env=None, config = None):
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
    
    def get_state(self, observation):
        if isinstance(self.env.observation_space, gym.spaces.Box):
            state = self.discretize_observation(observation)
        else:
            state = tuple(np.array(observation, dtype=int).flatten())

        return state
    
    def set_environment(self, environment):
        self.environment = environment
    