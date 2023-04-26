import os
import imageio

import streamlit as st
import subprocess
import importlib
import pytest
import traceback
import io
import sys
from src.utils import demonstrations
import altair as alt
from src.integration import comet
from src.agents.random_agent import RandomAgent
from src.agents.ql_agent import QLAgent
import numpy
import gymnasium as gym

st.set_page_config(layout="wide")

def main():
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page:", ["Home", "Testing", "Agent and Environment Descriptions", "Demonstrations", "Interactive Environment", "About", "Gallery"])

    if page == "Home":
        display_home()
    elif page == "Testing":
        display_testing()
    elif page == "Agent and Environment Descriptions":
        display_descriptions()
    elif page == "Demonstrations":
        display_demonstrations()
    elif page == "Interactive Environment":
        display_agent_environment()
    elif page == "Gallery":
        display_gallery()
    elif page == "About":
        display_about()
 
def display_home():
    page_bg_img = '''
    <style>
    body {
    background-image: url("https://media.giphy.com/media/YnexM9LwlwGu4Z1QnS/giphy.gif");
    background-size: cover;
    }
    </style>
    '''

    st.markdown(page_bg_img, unsafe_allow_html=True)

    st.header("Welcome to the Reinforcement Learning Gym Demo!")

    st.markdown("""
    This application demonstrates various reinforcement learning algorithms on different environments. 
    Key features include:
    
    - Interactive testing and visualization of reinforcement learning agents
    - Agent and environment descriptions
    - Demonstrations of pre-trained agents
    - Interactive environment with agent and environment selection
    """)


def display_testing():
    st.header("Testing Page")

    # List available test files
    test_files = [f for f in os.listdir("src/tests") if f.startswith("test_") and f.endswith(".py")]

    # Create a radio button for users to choose between running all tests or specific tests
    test_option = st.radio("Choose a testing option:", ["Run all tests", "Run specific tests"])

    # Create a multiselect widget for users to select test files if they chose "Run specific tests"
    if test_option == "Run specific tests":
        selected_test_files = st.multiselect("Select test files to run:", test_files)
    else:
        selected_test_files = test_files

    # Create a button to run the selected tests
    if st.button("Run tests"):
        if not selected_test_files:
            st.warning("No test files selected.")
            return

        # Capture the standard output to display test results
        stdout_capture = io.StringIO()
        sys.stdout = stdout_capture

        # Add the full path to the test files
        full_test_file_paths = [os.path.join("src/tests", test_file) for test_file in selected_test_files]

        # Run pytest programmatically
        pytest.main(full_test_file_paths)

        # Restore the standard output and display test results
        sys.stdout = sys.__stdout__
        test_results = stdout_capture.getvalue()
        st.write("```\n" + test_results + "\n```")

@st.cache_data
def load_app_text():
    with open(os.path.join("src", "integration", "descriptions.md"), "r") as f:
        app_text = f.read()
    return app_text

def display_descriptions():
    st.header("Agent and Environment Descriptions")
    app_text = load_app_text()
    st.markdown(app_text)

def display_demonstrations():
    st.header("Demonstrations")

    demo_names = demonstrations.selector()

    selected_demo = st.selectbox("Choose your demo:", demo_names)
    
    run_button = st.button("Run", key="run_demo")

    if run_button:
        if selected_demo:
            if selected_demo.lower() == "run all demos":
                demo_func = demonstrations.demo_all
            else:
                demo_func_name = f"demo_{selected_demo.lower()}"
                demo_func = getattr(demonstrations, demo_func_name)

            st.write(f"Running demo {selected_demo}...")
            try:
                demo_func()
                st.write(f"Demo {selected_demo} completed successfully.")
            except Exception as e:
                st.write(f"Demo {selected_demo} failed with error: {e}")
        else:
            st.write("Please select a demo to run.")

def create_agent_environment(agent_name, env_name):
    
    environment_instance = gym.make(env_name, render_mode = "rgb_array")

    if agent_name == "RandomAgent":
        agent_instance = RandomAgent(environment_instance)
    elif agent_name == "QLearningAgent":
        agent_instance = QLAgent(environment_instance)

    return agent_instance, environment_instance


def display_agent_environment():
    st.header("Select an agent and an environment")

    agent_names = ["RandomAgent", "QLearningAgent"]
    env_names = ["LunarLander-v2", "CartPole-v1", "Blackjack-v1", "MountainCarContinuous-v0"]

    agent_name = st.selectbox("Select an agent:", agent_names, key="agent")
    env_name = st.selectbox("Select an environment:", env_names, key="env")

    agent_instance, environment_instance = create_agent_environment(agent_name, env_name)

    if st.button("Run"):
        try:
            #probably use another function instead of hardcoding the loops here
            
            
            agent_instance.set_environment(environment_instance)
            num_episodes = 50  # set number of episodes manually
            max_steps_per_episode = 500

            frames = []
            for episode in range(num_episodes):
                state = environment_instance.reset()
                done = False
                total_reward = 0
                step = 0

                while not done:
                    action = agent_instance.choose_action(state)
                    observation, reward, terminated, truncated, info = environment_instance.step(action)
                    agent_instance.learn((state, action, reward, observation, done))
                    done = terminated or truncated
                    state = observation
                    total_reward += reward
                    step += 1

                    # Render the environment
                    frame = environment_instance.render()
                    frames.append(frame)

            # Save the frames as a video file
            video_file = f"visualizations/videos/{agent_name}_{env_name}_video.mp4"
            with imageio.get_writer(video_file, fps=30) as writer:
                for frame in frames:
                    writer.append_data(frame)

            # Display the video file
            st.video(video_file)

        except Exception as e:
            st.write(f"Interactive environment visualization failed with error: {e}")

def display_gallery():
    st.header("Gallery")
    st.write("Discover the performance of various agents in different environments.")
    
    # Add images, plots, or videos of agent performances
    # ...
    
    
def display_about():
    st.header("About")
    st.markdown("""
    Author: Nouri Mabrouk
    
    This project is part of a Master thesis in reinforcement learning.
    
    [Link to GitHub repo](https://github.com/Nourimabrouk/reinforcement-learning)
    """)

if __name__ == "__main__":
    main()