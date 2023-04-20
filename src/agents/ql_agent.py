import numpy as np
import gym
from gym import spaces
from .base_agent import BaseAgent

class QLAgent(BaseAgent):
    def __init__(self, env, config=None):
        super().__init__(config)
        self.env = env
        
        if self.config is None:
            self.config = {}

        self.config.setdefault("agent", {})
        self.config.setdefault("action_space", gym.spaces.Discrete(4))

        obs_space_shape = tuple(space.n for space in self.env.observation_space.spaces)
        self.q_table = np.zeros(obs_space_shape + (self.config["action_space"].n,))

        self.learning_rate = self.config["agent"].get("learning_rate", 0.1)
        self.discount_factor = self.config["agent"].get("discount_factor", 0.99)
        self.epsilon = self.config["agent"].get("epsilon", 0.1)
        self.epsilon_decay = self.config["agent"].get("epsilon_decay", 0.995)
        self.min_epsilon = self.config["agent"].get("min_epsilon", 0.01)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])

    def learn(self, transition):
        observation, action, reward, next_observation, done = transition
        state = self.get_state(observation)
        next_state = self.get_state(next_observation)

        target = reward + self.discount_factor * np.max(self.q_table[next_state]) * (not done)
        self.q_table[state][action] += self.learning_rate * (target - self.q_table[state][action])

        if done:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)
            
    def save(self, filepath):
        np.save(filepath, self.q_table)

    def load(self, filepath):
        self.q_table = np.load(filepath)

    def get_state(self, observation):
        state = tuple(np.array(observation, dtype=int).flatten())
        return state