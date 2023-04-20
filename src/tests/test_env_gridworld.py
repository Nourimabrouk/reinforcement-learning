import pytest
from src.environments.gridworld_env import GridWorld

@pytest.fixture
def custom_env():
    return GridWorld()

def test_reset(custom_env):
    observation = custom_env.reset()
    assert observation == (0, 0)

def test_step(custom_env):
    state = custom_env.reset()
    action = custom_env.action_space.sample()
    
    observation, reward, terminated, truncated, info = custom_env.step(action)
    done = terminated or truncated
    
    assert isinstance(observation, tuple)
    assert len(observation) == 2
    assert isinstance(reward, int)
    assert isinstance(done, bool)