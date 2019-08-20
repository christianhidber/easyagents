import unittest
import gym
from easyagents.env import _is_registered_with_gym, register_with_gym


class EnvTest(unittest.TestCase):
    def test_is_registered_with_gym(self):
        assert _is_registered_with_gym('CartPole-v0')

    def test_is_registered_with_gym_missingRegistration(self):
        assert not _is_registered_with_gym('MyEnv-v0')


class TestShimEnv(unittest.TestCase):

    def test_register_once(self):
        register_with_gym("test_env-v0", _Env1)
        env1 = gym.make("test_env-v0")
        assert isinstance(env1.unwrapped, _Env1)

    def test_register_twice(self):
        register_with_gym("test_env-v0", _Env1)
        register_with_gym("test_env-v0", _Env2)
        env2 = gym.make("test_env-v0")
        assert isinstance(env2.unwrapped, _Env2)

    def test_gym(self):
        gym.envs.registration.register(id="test_env-v1", entry_point=_Env1)
        gym.make("test_env-v1")


class _Env1(gym.Env):
    def __init__(self):
        pass

    def render(self, mode='human'):
        pass

    def reset(self):
        pass

    def step(self, action):
        pass


class _Env2(gym.Env):
    def __init__(self):
        pass

    def render(self, mode='human'):
        pass

    def reset(self):
        pass

    def step(self, action):
        pass


if __name__ == '__main__':
    unittest.main()
