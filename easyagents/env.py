"""This module contains support classes and methods to interact with OpenAI gym environments

    see https://github.com/openai/gym
"""

import gym.envs
import gym.error


def _is_registered_with_gym(gym_env_name: str) -> bool:
    """Determines if a gym environment with the name id exists.
    
        Args:
            gym_env_name: gym id to test.
            
        Returns:
            True if it exists, false otherwise
    """

    result = False
    try:
        spec = gym.envs.registration.spec(gym_env_name)
        assert spec is not None
        result = True
    except gym.error.UnregisteredEnv:
        pass
    return result
