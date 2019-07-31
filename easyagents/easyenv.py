import gym
import logging

"""
    this module is a hack a needs a fundamental rework and redesign (chh/19Q2)
"""


def register(gym_env_name: str = None, log_api: bool = True, log_steps: bool = False, log_reset: bool = False):
    """ Registers the EasyEnv wrapper for the 'gym_env_name' environment.
        The wrapper is registered as 'Easy-<env_name>'.
        max_episode_steps and reward_threshold is set according to the spec of the passed environment

        Limitation: currently only 1 gym env may be registered.

        Args:
        gym_env_name    the name of the gym environment to be wrapped by LogEnv (instead of gym_factor)
        log_api         if set all calls to the gym api are logged
        log_steps       if set to False calls to the step method are not logged (even if log_api is set)
        log_reset       if set to false calls to the reset method are not logged (even if log_api is set), 
                        except if the current episode is not complete yet.

        Retuns:
        The new name of the wrapped environment.
    """
    assert gym_env_name is not None, "None is not an admissible environment name"
    assert type(gym_env_name) is str, "gym_env_name is not a str"
    assert len(gym_env_name) > 0, "empty string is not an admissible environment name"

    result = EasyEnv.NAME_PREFIX + gym_env_name
    EasyEnv._log_steps = log_steps
    EasyEnv._log_reset = log_reset
    EasyEnv._log_api = log_api
    if EasyEnv._gym_env_name != gym_env_name:
        assert EasyEnv._gym_env_name is None, "Another environment was already registered"

        EasyEnv._gym_env_name = gym_env_name
        gym_spec = gym.envs.registration.spec(gym_env_name)
        gym.envs.registration.register(id=result,
                                       entry_point=EasyEnv,
                                       max_episode_steps=gym_spec.max_episode_steps,
                                       max_episode_seconds=gym_spec.max_episode_seconds,
                                       reward_threshold=gym_spec.reward_threshold)
    return result


class EasyEnv(gym.Env):
    """ Decorator for gym environments to intercept each gym env method call.
        The decorator is used to support method call logging, supporting callbacks
        and caching the last observation.
    """
    _gym_env_name = None
    _log_steps = False
    _log_reset = False
    _log_api = False
    _instanceCount = 0

    NAME_PREFIX = "Easy_"

    def __init__(self):
        target_env = gym.make(EasyEnv._gym_env_name)
        self.env = target_env.unwrapped
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata

        self._logStarted = False
        self._stepCount = 0
        self._totalStepCount = 0
        self._resetCount = 0
        self._renderCount = 0
        self._seedCount = 0
        self._closeCount = 0
        self._totalReward = 0.0
        self._done = False
        self._instanceId = EasyEnv._instanceCount
        EasyEnv._instanceCount += 1

        self._step_callback = None

        self._log = logging.getLogger(__name__)
        self._log.setLevel(logging.DEBUG)
        return

    def _log_api_call(self, msg):
        if self._log_api:
            if not self._logStarted:
                self._log.debug(f'#EnvId ResetCount.Steps [R=sumRewards]')
                self._logStarted = True
            logMsg = f'#{self._instanceId} {self._resetCount:3}.{self._stepCount:<3} [totalReward={self._totalReward:6.1f}] {msg}'
            self._log.debug(logMsg)
        return

    def _set_step_callback(self, callback):
        ''' callback is called after each execution of the step method.

            signature: callback(gym_env,action,state,reward,step,done,info)
        '''
        self._step_callback = callback

    def step(self, action):
        self._stepCount += 1
        self._totalStepCount += 1
        result = self.env.step(action)
        (state, reward, done, info) = result
        self._totalReward += reward
        if self._log_steps:
            self._log_api_call(
                f'executing step( {action} ) = ( reward={reward}, state={state}, done={done}, info={info} )')
        if done:
            self._log_api_call(f'game over')
            self._done = True
        if self._step_callback:
            self._step_callback(gym_env=self.env, action=action, state=state, reward=reward,
                                step=self._stepCount, done=done, info=info)
        return result

    def reset(self, **kwargs):
        if self._log_reset or not self._done:
            msg = "executing reset(...)"
            if not self._done and self._stepCount > 0:
                msg += " [episode not done]"
            self._log_api_call(msg)
        self._resetCount += 1
        self._stepCount = 0
        self._totalReward = 0.0
        self._done = False
        result = self.env.reset(**kwargs)
        if self._step_callback:
            self._step_callback(gym_env=self.env, action=None, state=result, reward=None,
                                step=0, done=False, info=None)
        return result

    def render(self, mode='human', **kwargs):
        self._log_api_call("executing render(...)")
        self._renderCount += 1
        return self.env.render(mode, **kwargs)

    def close(self):
        if self.env:
            self._log_api_call("executing close()")
            self._closeCount += 1
            return self.env.close()

    def seed(self, seed=None):
        self._log_api_call("executing seed(...)")
        self._seedCount += 1
        return self.env.seed(seed)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def spec(self):
        return self.env.spec
