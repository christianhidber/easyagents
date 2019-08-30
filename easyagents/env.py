"""This module contains support classes and methods to interact with OpenAI gym environments

    see https://github.com/openai/gym
"""
import inspect
import numpy as np
import gym.envs
import gym.error
import gym.spaces
import gym
import sys


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
def register_with_gym(gym_env_name: str, entry_point: type, max_episode_steps: int = 100000):
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


class _StepCountEnv(gym.core.Env):
    """Debug Env that runs forever, counting the calls to reset and step."""

    metadata = {'render.modes': ['ansi']}
    reward_range = (0, 1)
    max = 10^10
    action_space = gym.spaces.Box(low=-max,high=max,shape=(1,))
    observation_space = gym.spaces.Box(low=--max,high=max,shape=(1,))

    step_count : int = 0
    reset_count : int = 0

    @staticmethod
    def register_with_gym():
        """Register this environment with gym and yields the gym environment name."""
        result = "_StepCountEnv-v0"
        register_with_gym(result, _StepCountEnv)
        return result

    @staticmethod
    def reset_counts():
        _StepCountEnv.step_count = 0
        _StepCountEnv.reset_count = 0

    def __str__(self):
        return f'reset_count={_StepCountEnv.reset_count} step_count={_StepCountEnv.step_count}'

    def step(self, action):
        _StepCountEnv.step_count +=1
        return (_StepCountEnv.step_count, 1, False, None)

    def reset(self):
        _StepCountEnv.reset_count=0

    def render(self, mode='ansi'):
        return str(self)