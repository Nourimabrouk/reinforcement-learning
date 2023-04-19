from agents.base_agent import BaseAgent

class RandomAgent(BaseAgent):
    def choose_action(self, state):
        return self.config["action_space"].sample()

    def learn(self, transition):
        pass