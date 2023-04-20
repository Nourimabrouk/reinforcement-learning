def get_available_environments():
    from .base_environment import BaseEnvironment
    from .custom_grid_world import CustomGridWorld

    return [BaseEnvironment.__name__, CustomGridWorld.__name__]