from comet_ml import Experiment
import streamlit as st
import plotly.graph_objects as go

api_key = "rmUirVtt14dtLV0tBScUMz9fL"
project_name = "reinforcement-learning"
workspace = "nourimabrouk"

def initialize_experiment(api_key="yrmUirVtt14dtLV0tBScUMz9fL", project_name="reinforcement-learning"):
    experiment = Experiment(api_key=api_key, project_name=project_name)
    return experiment

def run_experiment(experiment, env, agent, experiment_name, render_video, num_episodes=5000):
    log_parameters(experiment, {"experiment": experiment_name})
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        video_frames = []

        while not done:
            action = agent.choose_action(state)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            transition = (state, action, reward, observation, done)
            agent.learn(transition)
            state = observation
            total_reward += reward

            if render_video:
                frame = env.render(mode='rgb_array')
                video_frames.append(frame)

        metrics = {"episode_reward": total_reward}
        log_metrics(experiment, metrics)
        log_episode(experiment, episode, metrics, video_frames)


def log_hyperparameters(experiment, hyperparameters):
    experiment.log_parameters(hyperparameters)

def log_parameters(experiment, parameters):
    experiment.log_parameters(parameters)

def log_metrics(experiment, metrics):
    experiment.log_metrics(metrics)

def log_episode(experiment, episode, metrics):
    for metric_name, metric_value in metrics.items():
        experiment.log_metric(metric_name, metric_value, step=episode)

def upload_model_weights(experiment, model_filepath):
    experiment.log_asset(model_filepath)

def end_experiment(experiment):
    experiment.end()

def get_experiment_data(experiment_key):
    return Experiment(api_key=api_key, project_name=project_name, previous_experiment=experiment_key)

def plot_experiment_metrics(experiment_data):
    fig = go.Figure()

    for metric_name in experiment_data.get_metrics_summary():
        data = experiment_data.get_metric(metric_name)
        fig.add_trace(go.Scatter(x=list(range(len(data))), y=data, mode='lines', name=metric_name))

    fig.update_layout(title='Experiment Metrics', xaxis_title='Step', yaxis_title='Value')
    fig.show()

def streamlit_comet_interaction(api_key, project_name):
    st.title("Comet.ml Experiment Interaction")
    
    # Create a Comet experiment instance
    experiment = Experiment(api_key=api_key, project_name=project_name, workspace=workspace)

    # Fetch the list of experiments in the project
    experiments = experiment.get_all()

    experiment_options = [(exp.id, f"{exp.name} ({exp.id})") for exp in experiments]
    
    # Create a dropdown to select an experiment
    selected_experiment_id, selected_experiment_label = st.selectbox(
        "Select an Experiment",
        options=experiment_options,
        format_func=lambda x: x[1]
    )

    if selected_experiment_id:
        # Load the experiment
        experiment = get_experiment_data(selected_experiment_id)

        # Display experiment details
        st.write(f"Experiment: {experiment.name}")
        st.write(f"Tags: {', '.join(experiment.get_tags())}")

        # Display experiment metrics
        metrics = experiment.get_metrics()
        st.write("Metrics:")
        for metric in metrics:
            st.write(f"{metric['name']}: {metric['valueMax']}")

        # Display experiment hyperparameters
        hyperparameters = experiment.get_parameters()
        st.write("Hyperparameters:")
        for key, value in hyperparameters.items():
            st.write(f"{key}: {value}")

        # Display any other experiment-related information, plots, or images
        # ...

    else:
        st.write("No experiments available.")