import os
import streamlit as st
import subprocess
import importlib
import pytest
import io
import sys
from src.utils import demonstrations
import altair as alt
from src.integration import comet
from src.agents.random_agent import RandomAgent
from src.agents.ql_agent import QLAgent
import numpy
import gymnasium as gym

def main():
    st.set_page_config(layout="wide")
    
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
    elif page == "Interactive Environment":
        display_interactive_environment()   
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
    test_files = [f for f in os.listdir("tests") if f.startswith("test_") and f.endswith(".py")]

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
        full_test_file_paths = [os.path.join("tests", test_file) for test_file in selected_test_files]

        # Run pytest programmatically
        pytest.main(full_test_file_paths)

        # Restore the standard output and display test results
        sys.stdout = sys.__stdout__
        test_results = stdout_capture.getvalue()
        st.write("```\n" + test_results + "\n```")

@st.cache
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

    # Get a list of all the demo functions in the current module
    demo_names = demonstrations.selector()

    # Create a dropdown menu to select a demonstration
    selected_demo = st.selectbox("Select a demonstration:", demo_names)

    # Provide description of the selected demonstration (markdown)
    demo_description = f"**{selected_demo}**: A demonstration of {selected_demo}."
    st.markdown(demo_description)

    if st.button("Run demonstration"):
        # Get the selected demo function
        demo_func = getattr(demonstrations, f"demo_{selected_demo.lower()}")

        # Run the selected demo function and capture the output
        output = io.StringIO()
        sys.stdout = output
        demo_func()
        sys.stdout = sys.__stdout__

        # Display the output in the Streamlit app
        st.text(output.getvalue())

@st.cache(allow_output_mutation=True)
def create_agent_environment(agent_name, env_name):
    if agent_name == "RandomAgent":
        agent_instance = RandomAgent()
    elif agent_name == "QLearningAgent":
        agent_instance = QLAgent()

    environment_instance = gym.make(env_name)

    return agent_instance, environment_instance

def display_agent_environment():
    st.header("Select an agent and an environment")

    agent_names = ["RandomAgent", "QLearningAgent"]
    env_names = ["LunarLander-v2", "CartPole-v1"]

    agent_name = st.sidebar.selectbox("Select an agent:", agent_names)
    env_name = st.sidebar.selectbox("Select an environment:", env_names)

    agent_instance, environment_instance = create_agent_environment(agent_name, env_name)
    display_interactive_environment(agent_instance, environment_instance)

def display_interactive_environment(agent, environment):
    st.header("Interactive Environment Visualization")

    experiment = comet.initialise_experiment()
    experiment.log_parameter("Agent", str(agent))
    experiment.log_parameter("Environment", str(environment))

    agent.set_environment(environment)
    for episode in range(environment.num_episodes):
        state = environment.reset()
        episode_reward = 0

        for step in range(environment.max_steps_per_episode):
            action = agent.choose_action(state)
            next_state, reward, truncated, terminated, info = environment.step(action)
            done = truncated or terminated
            episode_reward += reward

            agent.learn((next_state, reward, truncated, terminated, info))

            if done:
                break

            state = next_state

        experiment.log_metric("Episode Reward", episode_reward, step=episode)

    experiment.end()

    environment.render_interactive()

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