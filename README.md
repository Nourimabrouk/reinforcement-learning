# Reinforcement Learning Project

The Reinforcement Learning Project is a modular Python project aimed at building and comparative analysis of various reinforcement learning agents and environments.

## Key Features
Experimentation with a variety of agents and environment
Streamlit app deployment
CometML integration.

## Installation

To install the project, clone the repository and create a virtual environment using the `env` directory. Then, activate the virtual environment and install the required packages using the following command:

pip install -r requirements.txt

The project requires Python version 3.11.3 or later.

## Usage

The entry point of the project is `main.py`, which installs required packages, imports all modules in `src` and `configs`, and prints a simple output. To run the project, simply run the following command:

python main.py 

The project structure is as follows:
reinforcementlearning/
            ├── env/
            │   ├── etc/
            │   ├── Lib/
            │   ├── Scripts/
            │   └── ...
            ├── src/
            │   ├── agents/
            │   │   ├── __init__.py
            │   │   ├── base_agent.py
            │   │   └── ...
            │   ├── environments/
            │   │   ├── __init__.py
            │   │   ├── base_environment.py
            │   │   └── ...
            │   ├── utils/
            │   │   ├── __init__.py
            │   │   ├── visualization.py
            │   │   ├── logging.py
            │   │   └── ...
            │   ├── tests/
            │   │   ├── __init__.py
            │   │   ├── base_test.py
            │   │   └── ...
            │   ├── configs/
            │   │   ├── __init__.py
            │   │   ├── config.py
            │   │   └── ...
            │   ├── integrations/
            │   │   ├── __init__.py
            │   │   ├── comet_ml.py
            │   │   ├── streamlit_app.py
            │   │   └── ...
            │   └── __init__.py
            └── main.py
            
## License

This project is released under the MIT License.

## Contact Information

If you have any questions or feedback, please contact Nouri Mabrouk at nouri.mabrouk@gmail.com.