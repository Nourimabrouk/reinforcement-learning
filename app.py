import os
import streamlit as st
import subprocess
import importlib
import pytest
import io
import sys
from src.utils import demonstrations
import altair as alt


# Setup web deployment:
# subprocess.call(["pip", "install", "-r", "requirements.txt"])
# from src.integration.comet_ml import CometIntegration
# comet_integration = CometIntegration()

def main():
    st.set_page_config(layout="wide")
    col1, col2 = st.columns(2)

    with col1:
        st.title("Reinforcement Learning Gym Demo")
        st.header("Welcome to the Interactive RL Demo!")
        st.write("Explore various reinforcement learning agents and environments, run tests, and watch demonstrations using Comet ML for tracking and visualization.")

    with col2:
        st.title("Navigation")
        page = st.radio("Select a page:", ["Home", "Testing", "Agent and Environment Descriptions", "Demonstrations", "About", "Gallery"])

    if page == "Home":
        display_home()
    elif page == "Testing":
        display_testing()
    elif page == "Agent and Environment Descriptions":
        display_descriptions()
    elif page == "Demonstrations":
        display_demonstrations()
    elif page == "About":
        display_about()
    elif page == "Gallery":
        display_gallery()

def display_home():
# Set page background color
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
    """)

def display_testing():
    st.header("Testing Page")
    # Provide interactive testing environment using pytests
    # You will need to create interactive widgets for users to select the test files or specific tests to run.
    # Then run pytest programmatically and display the test results.

    # List available test files
    test_files = [f for f in os.listdir("tests") if f.startswith("test_") and f.endswith(".py")]

    # Create a multiselect widget for users to select test files
    selected_test_files = st.multiselect("Select test files to run:", test_files)

    # Create a button to run the selected tests
    if st.button("Run tests"):
        # Capture the standard output to display test results
        stdout_capture = io.StringIO()
        sys.stdout = stdout_capture

        # Run pytest programmatically
        pytest.main(selected_test_files)

        # Restore the standard output and display test results
        sys.stdout = sys.__stdout__
        test_results = stdout_capture.getvalue()
        st.write("```\n" + test_results + "\n```")

def display_descriptions():
    st.header("Agent and Environment Descriptions")
    # Load the text from app_text.md
    with open(os.path.join("src", "integration", "app_text.md"), "r") as f:
        app_text = f.read()
    # Display the text as formatted Markdown
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

def display_agent_environment():
    st.header("Select an agent and an environment")
    # Load the list of available agents and environments
    agent_names = ["RandomAgent", "QLearningAgent"]  # replace with actual list of agent names
    
    # Read environment names from src/environments/environments.md
    with open(os.path.join("src", "environments", "environments.md"), "r") as f:
        env_lines = f.readlines()
    env_names = [line.strip() for line in env_lines if line.startswith("-")]
    
    # Display dropdown menus to select an agent and an environment
    agent_name = st.selectbox("Select an agent:", agent_names)
    env_name = st.selectbox("Select an environment:", env_names)
    # Import the selected agent and environment modules
    agent_module = importlib.import_module(f"src.agents.{agent_name.lower()}")
    environment_module = importlib.import_module(f"src.environments.{env_name.lower()}")
    # Create instances of the agent and environment
    agent_instance = agent_module.Agent()
    environment_instance = environment_module.Environment()
    # Display the interactive environment visualization
    display_interactive_environment(agent_instance, environment_instance)

def display_interactive_environment(agent, environment):
    st.header("Interactive Environment Visualization")
    agent.set_environment(environment)
    agent.train()
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