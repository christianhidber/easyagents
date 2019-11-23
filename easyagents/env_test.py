import unittest
import gym
from easyagents.env import _is_registered_with_gym, register_with_gym, _LineWorldEnv

_line_world_name = _LineWorldEnv.register_with_gym()


class TestLineWorld(unittest.TestCase):
    def test_is_registered_with_gym(self):
        assert _is_registered_with_gym('LineWorld-v0')

    def test_make(self):
        lw = gym.make(_line_world_name)
        assert isinstance(lw.action_space, gym.spaces.Discrete)
        assert isinstance(lw.observation_space, gym.spaces.Box)

    def test_step_right_left(self):
        lw = gym.make(_line_world_name)
        obs = lw.reset()
        assert obs[0] == 3
        obs, reward, done, info = lw.step(1)
        assert obs[0] == 4
        assert reward == -1
        assert done == False
        obs, reward, done, info = lw.step(0)
        assert obs[0] == 3
        assert reward == 4
        assert done == False

    def test_step_left(self):
        lw = gym.make(_line_world_name)
        lw.reset()
        count = 0
        while True:
            o, r, d, i = lw.step(0)
            count += 1
            if d:
                break
        assert count == 3

    def test_step_right(self):
        lw = gym.make(_line_world_name)
        lw.reset()
        count = 0
        while True:
            o, r, d, i = lw.step(1)
            count += 1
            if d:
                break
        assert count == 37


class TestEnv(unittest.TestCase):
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
