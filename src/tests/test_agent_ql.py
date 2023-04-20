import pytest
import gymnasium as gym
import numpy as np
from src.agents.ql_agent import QLAgent
from src.environments.gridworld_env import GridWorld


@pytest.fixture
def ql_agent():
    env = GridWorld()
    agent = QLAgent(env)
    return agent


def test_choose_action(ql_agent):
    state = (0, 0)
    action = ql_agent.choose_action(state)
    assert 0 <= action < 4


def test_learn(ql_agent):
    initial_q_table = ql_agent.q_table.copy()
    transition = (ql_agent.env.reset(), 0, 1, ql_agent.env.step(0)[0], False)
    ql_agent.learn(transition)
    assert not np.array_equal(initial_q_table, ql_agent.q_table)