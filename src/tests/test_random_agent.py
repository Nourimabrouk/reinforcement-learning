import pytest
from src.agents.random_agent import RandomAgent
import gymnasium as gym

@pytest.fixture
def random_agent():
    config = {
        "action_space": gym.spaces.Discrete(4)
    }
    return RandomAgent(config)

def test_choose_action(random_agent):
    state = None
    action = random_agent.choose_action(state)
    assert random_agent.config["action_space"].contains(action)

def test_learn(random_agent):
    transition = (None, None, None, None, None)
    random_agent.learn(transition)  # Test if the learn function doesn't raise any exceptions