import logging
import unittest

import gym
from tf_agents.environments import suite_gym

import easyagents.easyenv
import easyagents.tfagents
from easyagents.config import LoggingVerbose
from easyagents.config import TrainingFast
from easyagents.config import TrainingSingleEpisode

logging.basicConfig(level=logging.DEBUG)


class TestEasyEnv(unittest.TestCase):
    step_callback_call_count = 0

    def test_gym_make(self):
        env_name = 'CartPole-v0'
        gym.make(env_name)

    def test_instance_counts(self):
        env_name = 'CartPole-v0'
        name = easyagents.easyenv._register(env_name)
        old_count = 0
        if name in easyagents.easyenv._EasyEnv._instance_counts:
            old_count = easyagents.easyenv._EasyEnv._instance_counts[name]
        gym.make(name)
        new_count = easyagents.easyenv._EasyEnv._instance_counts[name]
        assert new_count == (old_count + 1)

    def test_LoggingVerbose(self):
        ppo_agent = easyagents.tfagents.PpoAgent('CartPole-v0', training=TrainingFast(), logging=LoggingVerbose())
        ppo_agent.train()
        return

    def test_register_once(self):
        env_name = 'CartPole-v0'
        name = easyagents.easyenv._register(env_name)
        assert name == "Easy_" + env_name
        env = gym.make(name)
        assert env is not None

    def test_register_properties_set(self):
        env_name = 'CartPole-v0'
        spec = gym.envs.registration.spec(env_name)
        easyenv_name = easyagents.easyenv._register(env_name)
        easy_spec = gym.envs.registration.spec(easyenv_name)

        assert spec.max_episode_seconds == easy_spec.max_episode_seconds
        assert spec.max_episode_steps == easy_spec.max_episode_steps
        assert spec.reward_threshold == easy_spec.reward_threshold

    def test_register_tfagentsSuiteGymLoad(self):
        env_name = 'CartPole-v0'
        easyenv_name = easyagents.easyenv._register(env_name)
        tf_env = suite_gym.load(easyenv_name)
        assert tf_env is not None

    def test_register_twiceSameEnv(self):
        env_name = 'CartPole-v0'
        easyagents.easyenv._register(env_name)
        easyagents.easyenv._register(env_name)

    def test_register_twiceDifferentEnvs(self):
        env_name1 = 'CartPole-v0'
        n1 = easyagents.easyenv._register(env_name1)
        env_name2 = 'MountainCar-v0'
        n2 = easyagents.easyenv._register(env_name2)
        assert n1 != n2

    def test_set_step_callback(self):
        ppo_agent = easyagents.tfagents.PpoAgent('CartPole-v0', training=TrainingSingleEpisode(),
                                                 logging=LoggingVerbose())
        ppo_agent.train()

        TestEasyEnv.step_callback_call_count = 0
        (reward, steps) = ppo_agent.play_episode(callback=_step_callback)
        assert reward > 0
        assert steps > 0
        assert TestEasyEnv.step_callback_call_count > 0


def _step_callback(gym_env, action, state, reward, step, done, info):
    TestEasyEnv.step_callback_call_count += 1
    return


class TestShimEnv(unittest.TestCase):

    def test_register_once(self):
        easyagents.easyenv.register_with_gym("test_env-v0", _Env1)
        env1 = gym.make("test_env-v0")
        assert isinstance(env1.unwrapped, _Env1)

    def test_register_twice(self):
        easyagents.easyenv.register_with_gym("test_env-v0", _Env1)
        easyagents.easyenv.register_with_gym("test_env-v0", _Env2)
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
