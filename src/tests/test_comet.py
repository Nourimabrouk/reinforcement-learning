import pytest
import tempfile
import os

import plotly.graph_objs as go
from comet_ml import Experiment
from comet_ml import ExistingExperiment
from plotly.graph_objects import Figure

from src.integration.comet import (
    initialize_experiment,
    log_hyperparameters,
    log_metrics,
    log_episode,
    upload_model_weights,
    end_experiment,
    get_experiment_data,
    plot_experiment_metrics
)


@pytest.fixture
def experiment():
    api_key = "rmUirVtt14dtLV0tBScUMz9fL"
    project_name = "reinforcement-learning"
    workspace = "nourimabrouk"
    experiment = Experiment(api_key=api_key, project_name=project_name, workspace=workspace, disabled=True)
    return experiment


def test_initialize_experiment(experiment):
    api_key = "rmUirVtt14dtLV0tBScUMz9fL"
    project_name = "reinforcement-learning"

    exp = initialize_experiment(api_key=api_key, project_name=project_name)

    assert exp.api_key == api_key
    assert exp.project_name == project_name
    

