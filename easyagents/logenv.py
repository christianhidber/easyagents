import gym
import logging

def register(gym_env_name):
    """ Registers the LogEnv wrapper for the 'gym_env_name' environment.

        The wrapper is registered as 'Log-<env_name>'.
        The registered name is returned.
    """
    assert gym_env_name is not None, "None is not an admissible environment name"
    assert type(gym_env_name) is str , "gym_env_name is not a str"
    assert len(gym_env_name) > 0, "empty string is not an admissible environment name"

    result = "Log" + gym_env_name
    log_info("executing register({})".format(result),logging.getLogger(__name__))
    if LogEnv._gym_env_name != gym_env_name:
        assert LogEnv._gym_env_name is None, "Another environment was already registered"

        LogEnv._gym_env_name = gym_env_name
        gym.envs.registration.register(id=result, entry_point=LogEnv)
    return result

def log_info(msg, logger=None):
    if logger:
        logger.info(msg)
    else:
        print(msg)
    return

class LogEnv(gym.Env):
    """Decorator for gym environments to log each method call on the logger
    """
    _gym_env_name = None
    
    def __init__(self):
        target_env = gym.make( LogEnv._gym_env_name )
        self.env = target_env.unwrapped
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata

        self._stepCount=0
        self._totalStepCount=0
        self._resetCount=0
        self._renderCount=0
        self._seedCount=0
        self._closeCount=0

        self._log = logging.getLogger(__name__)
        self._log.setLevel(logging.DEBUG)
        return


    def _logCall(self, counter, msg):
        logMsg = f'{self._resetCount:3}.{self._stepCount:3} [{self._totalStepCount:3}] {msg}'
        log_info( logMsg, self._log )
        return

    def step(self, action):        
        self._stepCount += 1
        self._totalStepCount += 1
        result = self.env.step(action)
        (reward, state, done, info ) = result
        self._logCall(self._stepCount,f'step({action})=({reward},{state},{done},{info})' )
        if done:
            self._logCall(self._stepCount, "game over" )
        return result

    def reset(self, **kwargs):
        self._logCall(self._resetCount, "executing reset(...)" )
        self._resetCount += 1
        self._stepCount=0
        return self.env.reset(**kwargs)

    def render(self, mode='human', **kwargs):
        self._logCall(self._renderCount, "executing render(...)" )
        self._renderCount += 1
        return self.env.render(mode, **kwargs)

    def close(self):
        if self.env:
            self._logCall(self._closeCount, "executing close()" )
            self._closeCount += 1
            return self.env.close()

    def seed(self, seed=None):
        self._logCall(self._closeCount, "executing seed(...)" )
        self._seedCount += 1
        return self.env.seed(seed)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def spec(self):
        return self.env.spec
