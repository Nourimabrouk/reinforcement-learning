import os
import sys
from src.environments.gridworld_env import GridWorld
from src.agents.random_agent import RandomAgent
from PIL import Image
import gymnasium as gym
import gymnasium.wrappers as gw
import time
import streamlit as st
import subprocess
import inspect
import sys
import os
import sys
import inspect

from ..agents.random_agent import RandomAgent
from ..agents.ql_agent import QLAgent
from ..environments.gridworld_env import GridWorld
from ..integration.comet import initialize_experiment, run_experiment, log_parameters, log_hyperparameters, log_metrics, log_episode, upload_model_weights, end_experiment

RENDER_VIDEO = False
api_key = "rmUirVtt14dtLV0tBScUMz9fL"
project_name = "reinforcement-learning"
workspace = "nourimabrouk"

def log_episode(experiment, episode, metrics, video_frames):
    for metric_name, metric_value in metrics.items():
        experiment.log_metric(metric_name, metric_value, step=episode)

    if video_frames:
        images = [Image.fromarray(frame) for frame in video_frames]
        experiment.log_images(images, name="episode_video", step=episode)
        
def selector():
    # Get a list of all the demo functions in the current module
    members = inspect.getmembers(sys.modules[__name__])
    demo_funcs = [func for name, func in members if callable(func) and name.startswith("demo_")]
    demo_names = [func.__name__[5:].capitalize() for func in demo_funcs]

    # Add "Run all demos" to the beginning of the list
    demo_names.insert(0, "Run all demos")

    return demo_names

def demo_all():
    # Get a list of all the demo functions in the current module
    members = inspect.getmembers(sys.modules[__name__])
    demo_funcs = [func for name, func in members if callable(func) and name.startswith("demo_")]

    success = True
    for func in demo_funcs:
        demo_name = func.__name__[5:].capitalize()
        st.write(f"Running demo {demo_name}...")
        try:
            func()
            st.write(f"Demo {demo_name} completed successfully.")
        except Exception as e:
            st.write(f"Demo {demo_name} failed with error: {e}")
            success = False
            break
        
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
    env = gw.Monitor(env, './visualizations/videos', force=True)
    agent = RandomAgent()

    num_episodes = 10
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = observation
            total_reward += reward

        print(f"Episode {episode + 1}/{num_episodes}, Total reward: {total_reward}")

    env.close()
    gym.upload('./visualizations/videos', api_key='rmUirVtt14dtLV0tBScUMz9fL') 
    
def demo_comet():

    # Set your Comet ML API key
    api_key = "rmUirVtt14dtLV0tBScUMz9fL"

    # Initialize the Comet experiment
    experiment = initialize_experiment(api_key=api_key, project_name="reinforcement-learning")

    # Set the experiment name
    experiment.set_name("test-experiment")
    

    # Set up the environment and agent
    env = gym.make("CartPole-v1")
    agent = RandomAgent()

    # Log environment and agent details
    log_parameters(experiment, {"environment_name": "CartPole-v1", "agent": "RandomAgent"})

    # Log hyperparameters and other relevant information
    hyperparameters = {"learning_rate": 0.001, "epochs": 100}
    log_hyperparameters(experiment, hyperparameters)

    # Train the agent
    num_episodes = 500
    log_parameters(experiment, {"num_episodes": num_episodes})

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        step = 0

        while not done:
            action = agent.choose_action(state)
            observation, reward, terminated, truncated, info = env.step(action)
            agent.learn((state, action, reward, observation, done))
            done = terminated or truncated
            state = observation
            total_reward += reward
            step += 1

        # Log episode metrics
        metrics = {"episode_reward": total_reward, "episode_length": step}
        log_metrics(experiment, metrics)

        # Log episode visuals (e.g. reward distribution)
        experiment.log_histogram_3d([total_reward], name="reward_distribution", step=episode)

    # End the Comet experiment
    end_experiment(experiment)

    # Show the experiment URL
    print(f"View the experiment at {experiment.url}")

def demo_QL():
    # Set your Comet ML API key
    api_key = "rmUirVtt14dtLV0tBScUMz9fL"

    # Initialize the Comet experiment
    experiment = initialize_experiment(api_key=api_key, project_name="reinforcement-learning")

    # Set the experiment name
    experiment.set_name("QLAgent-experiments")

    # Set up the environments and agents
    env1 = gym.make("FrozenLake-v1")
    agent1 = QLAgent(env1)

    env2 = gym.make("CartPole-v1")
    agent2 = QLAgent(env2)

    env3 = gym.make("Acrobot-v1")
    agent3 = QLAgent(env3)
    
    RENDER_VIDEO = True

    # Run experiments
    run_experiment(experiment, env1, agent1, "FrozenLake-v1", RENDER_VIDEO)
    run_experiment(experiment, env2, agent2, "CartPole-v1", RENDER_VIDEO)
    run_experiment(experiment, env3, agent3, "Acrobot-v1", RENDER_VIDEO)

    # End the Comet experiment
    end_experiment(experiment)
    # End the Comet experiment
    end_experiment(experiment)   
        
if __name__ == "__main__":
    selector()