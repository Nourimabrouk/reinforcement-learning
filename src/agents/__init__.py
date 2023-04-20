def get_available_agents():
    from .base_agent import BaseAgent
    from .ql_agent import QLAgent
    from .random_agent import RandomAgent

    return [BaseAgent.__name__, QLAgent.__name__, RandomAgent.__name__]