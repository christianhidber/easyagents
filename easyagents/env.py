"""This module contains support classes and methods to interact with OpenAI gym environments
    as well as a gym env implementation for unit tests (linewordl)

    see https://github.com/openai/gym
"""
import gym, gym.envs, gym.error, gym.spaces
import inspect
import math
import matplotlib as plt
import numpy as np
from typing import List, Optional, Tuple, Dict, Callable, Any


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
def register_with_gym(gym_env_name: str, entry_point: type, max_episode_steps: int = 100000, **kwargs):
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
        kwargs: the args passed to the entry_point constructor call
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
    _ShimEnv._entry_points[gym_env_name] = (entry_point, kwargs)


class _LineWorldEnv(gym.Env):
    """Simple environment for fast unittest, registered as 'LineWorld-v0'

        * an agent lives in a finite linear world of uneven elements
        * at each moment it is in a certain position
        * initial position is the middle
        * some positions gain rewards, some don't
        * rewards are between 0 and 15
        * agent can either move left or right
        * objective: maximize total reward = sum(rewards) + sum(steps)
        * Cost per step: -1
        * Done Condition: agent is at pos 0 or total reward <= -20
    """

    @staticmethod
    def register_with_gym():
        """Register this environment with gym and yields the gym environment name."""
        result = "LineWorld-v0"
        register_with_gym(result, _LineWorldEnv)
        return result

    def __init__(self, world: Optional[List[int]] = None):
        """Creates the lineword, size and rewards are given by the world arg.

        Args:
            world: list of rewards to collect in each position of the lineworld.
        """
        if world is None:
            world = [10, 0, 0, 5, 0, 2, 15]
        assert world, "world must not be None or empty."
        self.world: np.array = np.array(world)
        number_of_actions: int = 2
        self.action_space: gym.spaces.Discrete = gym.spaces.Discrete(number_of_actions)
        self.size_of_world: int = len(world)
        self.max_reward: int = max(world)
        self.min_reward: int = min(world)

        # the environment's current state is described by the position of the agent and the remaining rewards
        self.observation_size: int = 1 + self.size_of_world
        low: np.array = np.full(self.observation_size, self.min_reward)
        high: np.array = np.full(self.observation_size, self.max_reward)
        self.observation_space: gym.spaces.Box = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.reward_range: Tuple[int, int] = (self.min_reward, self.max_reward)
        self.steps: int = 0
        self.done: bool = False
        self.pos: int = 0
        self._figure = None
        self.reset()

    def get_observation(self):
        return np.append([self.pos], self.remaining_rewards)

    def reset(self):
        self.total_reward = 0
        self.done = False
        self.pos = math.floor(len(self.world) / 2)
        self.steps = 0
        self.remaining_rewards = np.array(self.world, copy=True)
        return self.get_observation()

    def step(self, action):
        """perform action on this lineword.

        Args:
            action: 0 ==> move left, 1 ==> move right
        """
        if isinstance(action, np.ndarray):
            assert action.size == 1, "action of type numpy.array as invalid size"
            action = (int)(action)
        if isinstance(action, np.int32):
            action = (int)(action)
        assert isinstance(action, int)
        if action <= 0 and self.pos > 0:
            self.pos -= 1
        if action > 0 and self.pos < self.size_of_world - 1:
            self.pos += 1

        reward = self.remaining_rewards[self.pos] - 1
        self.total_reward += reward
        self.remaining_rewards[self.pos] = 0

        self.done = (self.pos == 0) or (self.total_reward <= -20)
        self.steps += 1

        observation = self.get_observation()
        info = None
        return observation, reward, self.done, info

    def _render_to_ansi(self):
        return f'position: {self.pos}, remaining rewards: {self.remaining_rewards},' + \
               f'total reward so far: {self.total_reward}, steps so far: {self.steps}, game done: {self.done}'

    def _render_to_figure(self):
        """ Renders the current state as a graph with matplotlib """
        if self._figure is not None:
            plt.close(self._figure)
        self._figure, ax = plt.subplots(1, figsize=(8, 4))
        ax.set_ylim(bottom=-1, top=self.max_reward + 1)
        x = np.arange(0, self.size_of_world, 1, dtype=np.uint8)
        y = self.remaining_rewards
        plt.plot([self.pos, self.pos], [0, 2], 'r^-')
        ax.scatter(x, y, s=75)
        self._figure.canvas.draw()
        return self._figure

    def _render_to_rgb(self):
        """ convert the output of render_to_figure to a rgb_array """
        self._render_to_figure()
        self._figure.canvas.draw()
        buf = self._figure.canvas.tostring_rgb()
        num_cols, num_rows = self._figure.canvas.get_width_height()
        plt.close(self._figure)
        self._figure = None
        result = np.fromstring(buf, dtype=np.uint8).reshape(num_rows, num_cols, 3)
        return result

    def render(self, mode='ansi'):
        if mode == 'ansi':
            return self._render_to_ansi()
        elif mode == 'human':
            return self._render_to_figure()
        elif mode == 'rgb_array':
            return self._render_to_rgb()
        else:
            super().render(mode=mode)


class _ShimEnv(gym.Wrapper):
    """Wrapper to redirect the instantiation of a gym environment to its current implementation.
    """

    _KWARG_GYM_NAME = "shimenv_gym_name"
    _entry_points: Dict[str, Tuple[Callable, Dict]] = {}

    def __init__(self, **kwargs):
        assert _ShimEnv._KWARG_GYM_NAME in kwargs, f'{_ShimEnv._KWARG_GYM_NAME} missing from kwargs'

        self._gym_env_name = kwargs[_ShimEnv._KWARG_GYM_NAME]
        entry_point, gym_kwargs = _ShimEnv._entry_points[self._gym_env_name]
        self._gym_env = entry_point(**gym_kwargs)
        super().__init__(self._gym_env)

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class _StepCountEnv(gym.core.Env):
    """Debug Env that runs forever, counting the calls to reset and step."""

    metadata = {'render.modes': ['ansi']}
    reward_range = (0, 1)
    max = 10 ** 7
    action_space = gym.spaces.discrete.Discrete(2)
    observation_space = gym.spaces.Box(low=0, high=max, shape=(1,))

    step_count: int = 0
    reset_count: int = 0

    @staticmethod
    def register_with_gym():
        """Register this environment with gym and yields the gym environment name."""
        result = "_StepCountEnv-v0"
        register_with_gym(result, _StepCountEnv)
        return result

    @staticmethod
    def clear():
        _StepCountEnv.step_count = 0
        _StepCountEnv.reset_count = 0

    def __str__(self):
        return f'reset_count={_StepCountEnv.reset_count} step_count={_StepCountEnv.step_count}'

    def step(self, action):
        _StepCountEnv.step_count += 1
        # noinspection PyRedundantParentheses
        return [_StepCountEnv.step_count], 1, False, None

    def reset(self):
        _StepCountEnv.reset_count += 1
        return [_StepCountEnv.step_count]

    def render(self, mode='ansi'):
        return str(self)
