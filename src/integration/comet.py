# comet_ml.py
from comet_ml import Experiment
import os

def create_comet_experiment(api_key=None, project_name=None, workspace=None):
    """
    Create a Comet ML experiment for logging metrics, parameters, and visualizations.

    :param api_key: (str) Your Comet ML API key.
    :param project_name: (str) Name of the project in Comet ML.
    :param workspace: (str) Name of the workspace in Comet ML.
    :return: (comet_ml.Experiment) The Comet ML experiment object.
    """

    if api_key is None:
        api_key = os.environ.get("COMET_API_KEY")

    if project_name is None:
        project_name = "reinforcement-learning"

    if workspace is None:
        workspace = "your_workspace_name"

    experiment = Experiment(
        api_key=api_key,
        project_name=project_name,
        workspace=workspace,
        auto_param_logging=True,
        auto_metric_logging=True,
        parse_args=False,
    )

    return experiment

#     # Log the agent's performance
#     def log_agent(self, agent, environment, metrics):
#         # ...

#     # Retrieve the agent's performance history
#     def get_agent_history(self, agent, environment):
#         # ...