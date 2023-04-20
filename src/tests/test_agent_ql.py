import pytest
import gymnasium as gym
from src.agents.ql_agent import QLAgent
import numpy as np

@pytest.fixture
def ql_agent():
    config = {
        "action_space": gym.spaces.Discrete(4),
        "observation_space": gym.spaces.Box(low=0, high=255, shape=(84, 84, 4), dtype=np.uint8)
    }
    return QLAgent(config)

def test_choose_action(ql_agent):
    state = np.zeros((84, 84, 4))
    action = ql_agent.choose_action(state)
    assert ql_agent.config["action_space"].contains(action)

def test_learn(ql_agent):
    transition = (np.zeros((84, 84, 4)), 0, 0, np.zeros((84, 84, 4)), False)
    ql_agent.learn(transition)  # Test if the learn function doesn't raise any exceptions