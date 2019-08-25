"""Contains the implementation of the gym environment monitor used to intercept all calls from
   a BackendAgent to the gym environment. The monitor is the basis for ApiCallbacks as well as
   the gym environment statistics like the number of episodes played or the number of steps
   taken during training.
"""

from typing import Dict, Optional
import threading

import gym.core
import gym.envs
import easyagents.backends.core as bcore


class _MonitorTotalCounts(object):
    """Keeps usage counts on a monitored gym environment.

    Attributes:
        gym_env_name: the id of the monitored gym environment
    """

    def __init__(self, gym_env_name):
        assert isinstance(gym_env_name, str), "gym_env_name not a str"
        assert gym_env_name, "gym_env_name empty"

        self._original_env_name = gym_env_name
        self.gym_env_name = _MonitorEnv._NAME_PREFIX + self._original_env_name
        self._instances_created: int = 0
        self._episodes_done: int = 0
        self._steps_done: int = 0

    def __str__(self):
        return f'[{self._original_env_name}] #instances={self.instances_created} ' + \
               f'#episodes={self.episodes_done} #steps={self.steps_done}'

    @property
    def instances_created(self):
        """the total number of instances created"""
        with _MonitorEnv._lock:
            return self._instances_created

    def instances_created_inc(self):
        """increments the total number of instances created"""
        with _MonitorEnv._lock:
            self._instances_created += 1
            return self._instances_created

    @property
    def episodes_done(self):
        """the total number of episodes completed (over all instances).

            A new episodes starts with a new instance or a call to reset.
            If step() was not called after the last reset, than the episode count is not incremented.
            Thus calling reset() multiple times, or calling reset right after instantiation increments
            the episode count at most by 1.
        """
        with _MonitorEnv._lock:
            return self._episodes_done

    def episodes_done_inc(self):
        """increments the total number of new episodes (over all instances).
        """
        with _MonitorEnv._lock:
            self._episodes_done += 1
            return self._episodes_done

    @property
    def steps_done(self):
        """the number of step completed (over all instances)"""
        with _MonitorEnv._lock:
            return self._steps_done

    def steps_done_inc(self):
        """increments the total number of step() calls (over all instances)"""
        with _MonitorEnv._lock:
            self._steps_done += 1
            return self._steps_done


class _MonitorEnv(gym.Wrapper):
    """Intercepts all calls between a BackendAgent and its gym environment.

    Attributes:
        gym_env_name: the name of the monitored gym env
        instance: unique id among all monitor instances for the given gym_env
        total: statistics over all monitor instances for the given gym_env
        episodes_done: the number of episodes completed in this instance (episode reached done or was resetted)
        episode_sum_of_rewards: the sum of rewards over all steps in the current episode
        is_episode_done: true if the current episode is done (and a new episode isn't started yet with reset())
        steps_done_in_episode: the number of steps completed in the current episode
        steps_done_in_instance: the total number of steps in this instance
    """

    _NAME_PREFIX = "_MonitorEnv_"
    _KWARG_GYM_ENV_NAME = "easyagents_gym_env_name"
    _monitor_total_counts: Dict[str, _MonitorTotalCounts] = dict()
    _lock = threading.Lock()
    _backend_agent = None

    @staticmethod
    def _register_backend_agent(backend_agent):
        """Registers backend_agent als callback target for all new monitored gym instances.

            All newly created MonitorEnv will call backend_agent upon step / reset calls.
        """
        with _MonitorEnv._lock:
            if _MonitorEnv._backend_agent is None or backend_agent is None:
                _MonitorEnv._backend_agent = backend_agent
            else:
                assert _MonitorEnv._backend_agent is backend_agent, "another backend agent is already registered"
                _MonitorEnv._backend_agent = backend_agent

    def __init__(self, **kwargs):
        assert _MonitorEnv._KWARG_GYM_ENV_NAME in kwargs, f'{_MonitorEnv._KWARG_GYM_ENV_NAME} missing from kwargs'

        self.gym_env_name = kwargs[_MonitorEnv._KWARG_GYM_ENV_NAME]
        with _MonitorEnv._lock:
            self.total = _MonitorEnv._monitor_total_counts[self.gym_env_name]
        self.steps_done_in_instance: int = 0
        self.episodes_done: int = 0
        self.steps_done_in_episode: int = 0
        self.episode_sum_of_rewards: float = 0
        self.is_episode_done: bool = False
        self.instance_id = self.total.instances_created
        self._backend_agent: Optional[bcore.BackendAgent] = _MonitorEnv._backend_agent

        if self._backend_agent:
            self._backend_agent._on_gym_init_begin()
        gym_env = gym.make(self.gym_env_name)
        super().__init__(gym_env)
        self.total.instances_created_inc()
        if self._backend_agent:
            self._backend_agent._on_gym_init_end(self.env)

    def reset(self, **kwargs):
        """performs a reset on the wrapped environment."""
        if self._backend_agent:
            self._backend_agent._on_gym_reset_begin(self.env, **kwargs)

        result = self.env.reset(**kwargs)
        if self.steps_done_in_episode > 0 and not self.is_episode_done:
            self.episodes_done += 1
            self.total.episodes_done_inc()
        self.is_episode_done = False
        self.episode_sum_of_rewards = 0
        self.steps_done_in_episode = 0

        if self._backend_agent:
            self._backend_agent._on_gym_reset_end(self.env, result, **kwargs)
        return result

    def step(self, action):
        """performs a step on the wrapped environment.

        Hint:
        o the step count is incremented before step_end()
        """
        if self._backend_agent:
            self._backend_agent._on_gym_step_begin(self.env, action)

        result = self.env.step(action)
        (state, reward, done, info) = result
        if not self.is_episode_done and done:
            self.is_episode_done = True
            self.episodes_done += 1
            self.total.episodes_done_inc()
        self.episode_sum_of_rewards += reward
        self.steps_done_in_episode += 1
        self.steps_done_in_instance += 1
        self.total.steps_done_inc()

        if self._backend_agent:
            self._backend_agent._on_gym_step_end(self.env, action, result)
        return result

    def __str__(self):
        return f'[{self.gym_env_name}#{self.instance_id:}] {self.episodes_done:3}#{self.steps_done_in_episode:<3} ' + \
               f'Σr={self.episode_sum_of_rewards:6.1f}'


def _get(env: gym.Env) -> _MonitorEnv:
    """extracts from env the underlying _MonitorEnv, returns None of not successful"""
    assert env, "env not set"
    result = None
    if isinstance(env, _MonitorEnv):
        result = env
    else:
        if isinstance(env, gym.core.Wrapper):
            result = _get(env.env)
    return result


def _register_gym_monitor(gym_env_name: str) -> _MonitorTotalCounts:
    """Registers the _MonitorEnv wrapper for the 'gym_env_name' environment.

    The wrapper is registered as '_MonitorEnv-<env_name>'.
    max_episode_steps and reward_threshold are set according to the spec of gym_env_name.
    If the same name is registered more than once, the second and all following calls are noOps.

    Args:
        gym_env_name: the gym id of the environment to be monitored

    Retuns:
        the count monitor of gym_env_name
    """
    assert gym_env_name, "gym_env_name must be a non-empty string"
    assert type(gym_env_name) is str, "gym_env_name is not a str"

    result: _MonitorTotalCounts
    with _MonitorEnv._lock:
        if gym_env_name not in _MonitorEnv._monitor_total_counts:
            result = _MonitorTotalCounts(gym_env_name)
            gym_spec = gym.envs.registration.spec(gym_env_name)
            gym.envs.registration.register(id=result.gym_env_name,
                                           entry_point=_MonitorEnv,
                                           max_episode_steps=gym_spec.max_episode_steps,
                                           max_episode_seconds=gym_spec.max_episode_seconds,
                                           reward_threshold=gym_spec.reward_threshold,
                                           kwargs={_MonitorEnv._KWARG_GYM_ENV_NAME: gym_env_name})
            _MonitorEnv._monitor_total_counts[gym_env_name] = result
        else:
            result = _MonitorEnv._monitor_total_counts[gym_env_name]
    return result
