import numpy as np
from agents.base_agent import BaseAgent

class QLAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)

        self.q_table = np.zeros((4, 4, self.config["action_space"].n))

        self.learning_rate = self.config["agent"]["learning_rate"]
        self.discount_factor = self.config["agent"]["discount_factor"]
        self.epsilon = self.config["agent"]["epsilon"]
        self.epsilon_decay = self.config["agent"]["epsilon_decay"]
        self.min_epsilon = self.config["agent"]["min_epsilon"]

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.config["action_space"].sample()
        else:
            return np.argmax(self.q_table[state])

    def learn(self, transition):
        state, action, reward, next_state, done = transition

        target = reward + self.discount_factor * np.max(self.q_table[next_state]) * (not done)
        self.q_table[state][action] += self.learning_rate * (target - self.q_table[state][action])

        if done:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

    def save(self, filepath):
        np.save(filepath, self.q_table)

    def load(self, filepath):
        self.q_table = np.load(filepath)