import pytest
import numpy as np
from src.agents.ql_agent import QLAgent
import gymnasium as gym

@pytest.fixture
def ql_agent():
    config = {
        "agent": {
            "learning_rate": 0.1,
            "discount_factor": 0.99,
            "epsilon": 1.0,
            "epsilon_decay": 0.99,
            "min_epsilon": 0.1
        },
        "action_space": gym.spaces.Discrete(4)
    }
    return QLAgent(config)

def test_choose_action(ql_agent):
    state = (0, 0)
    action = ql_agent.choose_action(state)
    assert ql_agent.config["action_space"].contains(action)

def test_learn(ql_agent):
    # Test if the learn function doesn't raise any exceptions
    transition = ((0, 0), 0, 1.0, (0, 1), False)
    ql_agent.learn(transition)

def test_save_and_load(ql_agent, tmp_path):
    # Test if the save and load functions work correctly
    q_table = np.random.rand(4, 4, ql_agent.config["action_space"].n)
    ql_agent.q_table = q_table

    filepath = tmp_path / "q_table.npy"
    ql_agent.save(filepath)

    loaded_q_table = np.load(filepath)
    assert np.array_equal(ql_agent.q_table, loaded_q_table)