import numpy as np
import gymnasium as gym

from .base_agent import BaseAgent

import numpy as np
import gym

from .base_agent import BaseAgent


class QLAgent(BaseAgent):
    def __init__(self, env=None, config=None):
        self.config = config or {}
        self.env = env
        if self.env is None:
            raise ValueError("Environment must be provided")
        self.config.setdefault("agent", {})
        self.config.setdefault("action_space", self.env.action_space)
        self.config.setdefault("n_bins", 10)
        if isinstance(self.env.observation_space, gym.spaces.Box):
            obs_space_shape = (self.config["n_bins"],)
        elif isinstance(self.env.observation_space, gym.spaces.Tuple) or isinstance(self.env.observation_space, gym.spaces.Dict):
            try:
                obs_space_shape = tuple(space.n for space in self.env.observation_space.spaces)
            except AttributeError:
                raise ValueError("Observation space is not iterable")
        elif isinstance(self.env.observation_space, gym.spaces.Discrete):
            obs_space_shape = (self.env.observation_space.n,)
        else:
            raise ValueError("Observation space type not supported")

        self.q_table = np.zeros(obs_space_shape + (self.config["action_space"].n,))
        self.learning_rate = self.config["agent"].get("learning_rate", 0.1)
        self.discount_factor = self.config["agent"].get("discount_factor", 0.99)
        self.epsilon = self.config["agent"].get("epsilon", 0.1)
        self.epsilon_decay = self.config["agent"].get("epsilon_decay", 0.995)
        self.min_epsilon = self.config["agent"].get("min_epsilon", 0.01)

        super().__init__(self.config)

        super().__init__(self.config)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            state_idx = tuple(map(int, state))
            return np.argmax(self.q_table[state_idx])

    def learn(self, transition):
        observation, action, reward, next_observation, done = transition
        state = self.get_state(observation)
        next_state = self.get_state(next_observation)

        target = reward + self.discount_factor * np.max(self.q_table[next_state]) * (1 - done)
        self.q_table[state][action] += self.learning_rate * (target - self.q_table[state][action])

        if done:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

    def save(self, filepath):
        np.save(filepath, self.q_table)

    def load(self, filepath):
        self.q_table = np.load(filepath)

    def get_state(self, observation):
        if isinstance(self.env.observation_space, gym.spaces.Box):
            bins = np.array([np.linspace(self.env.observation_space.low[i], self.env.observation_space.high[i], self.config["n_bins"]) for i in range(self.env.observation_space.shape[0])])
            state = tuple(np.digitize(observation[i], bins[i]) - 1 for i in range(self.env.observation_space.shape[0]))
        elif isinstance(self.env.observation_space, gym.spaces.Discrete):
            state = observation
        else:
            raise ValueError("Observation space type not supported")

        return state

    def set_environment(self, environment):
        self.env = environment

    def discretize_observation(self, observation):
        bins = [np.linspace(low, high, self.config["n_bins"] + 1)[1:-1] for low, high in zip(self.env.observation_space.low, self.env.observation_space.high)]
        state = tuple(int(np.digitize(feature, bin)) for feature, bin in zip(observation, bins))

        return state