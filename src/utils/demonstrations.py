import gymnasium as gym
import subprocess
import inspect
import sys
from ..agents.random_agent import RandomAgent
from ..environments.gridworld_env import GridWorld
from ..integration.comet import create_comet_experiment
from ..agents.ql_agent import QLAgent



def selector():
    # Lets the user select from available demos, and runs the selected demo.
    # Make sure the selector can handle other demo_xxx() functions

    # Get a list of all the demo functions in the current module
    members = inspect.getmembers(sys.modules[__name__])
    demo_funcs = [func for name, func in members if callable(func) and name.startswith("demo_")]
    demo_names = [func.__name__[5:].capitalize() for func in demo_funcs]

    print("Available demos:")
    for i, name in enumerate(demo_names, start=1):
        print(f"{i}. {name}")

    demo_choice = input("Enter the demo number or name you want to run: ")
    selected_func = None

    # Try to parse input as integer first
    try:
        demo_choice_int = int(demo_choice)
        if demo_choice_int in range(1, len(demo_funcs)+1):
            selected_func = demo_funcs[demo_choice_int-1]
    except ValueError:
        pass

    # If input was not an integer or not a valid demo number, try to match input as demo name
    if not selected_func:
        for func, name in zip(demo_funcs, demo_names):
            if name.lower() == demo_choice.lower():
                selected_func = func
                break

    if selected_func:
        selected_func()
    else:
        print("Invalid demo number or name. Please try again.")
        selector()
    return demo_names

        
def demo_tests():
    
    # Demonstrates the functionality and commands assosiated with the tests
    # Defaults to: pytest test_agent_random.py
    
    subprocess.run(["pytest", "test_random_agent.py"])
    
def demo_gym():
        
    env = gym.make("LunarLander-v2", render_mode='human')
    env.reset()
    total_reward = 0

    for _ in range(1000):
        env.render()
        action = env.action_space.sample()
        observation, info = env.reset(seed=123, options={})
        done = False

        while not done:
            action = env.action_space.sample()  # agent policy that uses the observation and info
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        env.close()
        
        total_reward += reward

        if done:
            print(f"Episode finished after {_ + 1} timesteps")
            print(f"Total reward: {total_reward}")
            break

    env.close()

def demo_custom_env():

    env = GridWorld()
    agent = RandomAgent()

    num_episodes = 10
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            total_reward += reward

        print(f"Episode {episode + 1}/{num_episodes}, Total reward: {total_reward}")

    env.render()
    env.close()   

def demo_comet():
    """
    A demonstration of using Comet ML for logging metrics and parameters in a reinforcement learning experiment.
    """

    # Set your Comet ML API key
    api_key = "your_comet_ml_api_key_here"

    # Create a Comet ML experiment
    experiment = create_comet_experiment(api_key=api_key)

    # Set up the environment and agent
    env = gym.make("CartPole-v0")
    agent = RandomAgent(env)

    # Log environment and agent details
    experiment.log_parameter("environment_name", "CartPole-v0")
    experiment.log_parameter("agent", "RandomAgent")

    # Train the agent
    num_episodes = 500
    experiment.log_parameter("num_episodes", num_episodes)

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            total_reward += reward
            steps += 1

        # Log episode metrics
        experiment.log_metric("episode_reward", total_reward, step=episode)
        experiment.log_metric("episode_length", steps, step=episode)

        # Log episode visuals (e.g. reward distribution)
        experiment.log_histogram_3d([total_reward], name="reward_distribution", step=episode)

    # Log other assets, such as the agent's Q-table
    if hasattr(agent, "q_table"):
        experiment.log_asset_data(agent.q_table, "q_table.pkl")

    # End the experiment
    experiment.end()
        
if __name__ == "__main__":
    selector()