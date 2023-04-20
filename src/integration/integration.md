from comet_ml import Experiment

# Create a new Comet.ml experiment
experiment = Experiment(api_key="YOUR_API_KEY", project_name="my_project")

# Log a scalar metric
experiment.log_metric("accuracy", 0.95)

# Log a hyperparameter
experiment.log_parameter("learning_rate", 0.001)

# Log an artifact such as a model or dataset
experiment.log_asset("my_model.h5")

# Log a video of the training process
experiment.log_video("training.mp4")

