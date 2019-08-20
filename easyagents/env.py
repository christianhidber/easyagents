"""This module contains support classes and methods to interact with OpenAI gym environments

    see https://github.com/openai/gym
"""
import inspect

import gym.envs
import gym.error
import gym


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


# noinspection DuplicatedCode
def register_with_gym(gym_env_name: str, entry_point: type, max_episode_steps: int = None):
    """Registers the class entry_point in gym by the name gym_env_name allowing overriding registrations.

    Thus different implementations of the same class (and the same name) maybe registered consecutively.
    The latest registrated version is used for instantiation.
    This facilitates developing an environment in a jupyter notebook without haveing to
    reregister a modified class under a new name.

    limitation: the max_episode_steps value of the first registration holds for all registrations
        with the same gym_env_name

    Args:
        gym_env_name: the gym environment name to be used as argument with gym.make
        max_episode_steps: all episodes end latest after this number of steps
        entry_point: the class to be registed with gym id gym_env_name
    """
    assert gym_env_name is not None, "None is not an admissible environment name"
    assert type(gym_env_name) is str, "gym_env_name is not a str"
    assert len(gym_env_name) > 0, "empty string is not an admissible environment name"
    assert inspect.isclass(entry_point), "entry_point not a class"
    assert issubclass(entry_point, gym.Env), "entry_point not a subclass of gym.Env"
    assert callable(entry_point), "entry_point not callable"

    if gym_env_name not in _ShimEnv._entry_points:
        gym.envs.registration.register(id=gym_env_name,
                                       entry_point=_ShimEnv,
                                       max_episode_steps=max_episode_steps,
                                       kwargs={_ShimEnv._KWARG_GYM_NAME: gym_env_name})
    _ShimEnv._entry_points[gym_env_name] = entry_point


class _ShimEnv(gym.Wrapper):
    """Wrapper to redirect the instantiation of a gym environment to its current implementation.
    """

    _KWARG_GYM_NAME = "shimenv_gym_name"
    _entry_points = {}

    def __init__(self, **kwargs):
        assert _ShimEnv._KWARG_GYM_NAME in kwargs, f'{_ShimEnv._KWARG_GYM_NAME} missing from kwargs'

        self._gym_env_name = kwargs[_ShimEnv._KWARG_GYM_NAME]
        entry_point = _ShimEnv._entry_points[self._gym_env_name]
        self._gym_env = entry_point()
        super().__init__(self._gym_env)

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
