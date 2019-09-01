import inspect
import logging

import gym


def _register(gym_env_name: str, log_api: bool = True, log_steps: bool = False, log_reset: bool = False):
    """Registers the EasyEnv wrapper for the 'gym_env_name' environment.

    The wrapper is registered as 'Easy-<env_name>'.
    max_episode_steps and reward_threshold are set according to the spec of the passed environment.
    If the same name is registered more than once, the second and all following calls are noOps.

    Args:
        gym_env_name: the name of the gym environment to be wrapped by LogEnv (instead of gym_factor)
        log_api: if set all calls to the gym api are logged
        log_steps: if set to False calls to the step method are not logged (even if log_api is set)
        log_reset: if set to false calls to the reset method are not logged (even if log_api is set),
            except if the current episode is not complete yet.

    Retuns:
        The new name of the wrapped environment.
    """
    assert gym_env_name is not None, "None is not an admissible environment name"
    assert type(gym_env_name) is str, "gym_env_name is not a str"
    assert len(gym_env_name) > 0, "empty string is not an admissible environment name"

    result = _EasyEnv._NAME_PREFIX + gym_env_name
    if result not in _EasyEnv._instance_counts:
        gym_spec = gym.envs.registration.spec(gym_env_name)
        gym.envs.registration.register(id=result,
                                       entry_point=_EasyEnv,
                                       max_episode_steps=gym_spec.max_episode_steps,
                                       max_episode_seconds=gym_spec.max_episode_seconds,
                                       reward_threshold=gym_spec.reward_threshold,
                                       kwargs={_EasyEnv._KWARG_GYM_NAME: gym_env_name,
                                               _EasyEnv._KWARG_LOG_STEPS: log_steps,
                                               _EasyEnv._KWARG_LOG_RESET: log_reset,
                                               _EasyEnv._KWARG_LOG_API: log_api})
        _EasyEnv._instance_counts[result] = 0
    return result


class _EasyEnv(gym.Wrapper):
    """Wrapper for gym environments to intercept each gym env method call.

    The wrapper is used to support method call logging, supporting callbacks
    and caching the last observation.
    """
    _instance_counts = dict()

    _NAME_PREFIX = "Easy_"
    _KWARG_GYM_NAME = "easyenv_gym_name"
    _KWARG_LOG_STEPS = "easyenv_log_steps"
    _KWARG_LOG_RESET = "easyenv_log_reset"
    _KWARG_LOG_API = "easyenv_log_api"

    def __init__(self, **kwargs):
        assert _EasyEnv._KWARG_GYM_NAME in kwargs, f'{_EasyEnv._KWARG_GYM_NAME} missing from kwargs'
        assert _EasyEnv._KWARG_LOG_API in kwargs, f'{_EasyEnv._KWARG_LOG_API} missing from kwargs'
        assert _EasyEnv._KWARG_LOG_RESET in kwargs, f'{_EasyEnv._KWARG_LOG_RESET} missing from kwargs'
        assert _EasyEnv._KWARG_LOG_STEPS in kwargs, f'{_EasyEnv._KWARG_LOG_STEPS} missing from kwargs'

        self._gym_env_name = kwargs[_EasyEnv._KWARG_GYM_NAME]
        super().__init__(gym.make(self._gym_env_name))
        self._log_api = kwargs[_EasyEnv._KWARG_LOG_API]
        self._log_reset = kwargs[_EasyEnv._KWARG_LOG_RESET]
        self._log_steps = kwargs[_EasyEnv._KWARG_LOG_STEPS]

        easyenv_name = _EasyEnv._NAME_PREFIX + self._gym_env_name
        self._instance_id = _EasyEnv._instance_counts[easyenv_name]
        _EasyEnv._instance_counts[easyenv_name] = self._instance_id + 1

        self._close_count = 0
        self._render_count = 0
        self._reset_count = 0
        self._seed_count = 0
        self._step_count = 0
        self._done = False

        self._log = logging.getLogger(__name__)
        self._log.setLevel(logging.DEBUG)
        self._log_started = False

        self._step_callback = None
        self._total_reward = 0.0
        self._total_step_count = 0

    def _log_api_call(self, msg):
        if self._log_api:
            if not self._log_started:
                self._log.debug(f'#EnvId ResetCount.Steps [R=sumRewards]')
                self._log_started = True
            logMsg = f'#{self._instance_id} {self._reset_count:3}.{self._step_count:<3} [totalReward={self._total_reward:6.1f}] {msg}'
            self._log.debug(logMsg)
        return

    def _set_step_callback(self, callback):
        """callback is called after each execution of the step method.

        signature: callback(gym_env,action,state,reward,step,done,info)
        """
        self._step_callback = callback

    def step(self, action):
        self._step_count += 1
        self._total_step_count += 1
        result = self.env.step(action)
        (state, reward, done, info) = result
        self._total_reward += reward
        if self._log_steps:
            self._log_api_call(
                f'executing step( {action} ) = ( reward={reward}, state={state}, done={done}, info={info} )')
        if done:
            self._log_api_call(f'game over')
            self._done = True
        if self._step_callback:
            self._step_callback(gym_env=self.env, action=action, state=state, reward=reward,
                                step=self._step_count, done=done, info=info)
        return result

    def reset(self, **kwargs):
        if self._log_reset or not self._done:
            msg = "executing reset(...)"
            if not self._done and self._step_count > 0:
                msg += " [episode not done]"
            self._log_api_call(msg)
        self._reset_count += 1
        self._step_count = 0
        self._total_reward = 0.0
        self._done = False
        result = self.env.reset(**kwargs)
        if self._step_callback:
            self._step_callback(gym_env=self.env, action=None, state=result, reward=None,
                                step=0, done=False, info=None)
        return result

    def render(self, mode='human', **kwargs):
        self._log_api_call("executing render(...)")
        self._render_count += 1
        return self.env.render(mode, **kwargs)

    def close(self):
        if self.env:
            self._log_api_call("executing close()")
            self._close_count += 1
            return self.env.close()

    def seed(self, seed=None):
        self._log_api_call("executing seed(...)")
        self._seed_count += 1
        return self.env.seed(seed)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def spec(self):
        return self.env.spec
