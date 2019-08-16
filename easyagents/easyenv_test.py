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

    def test_cartpole(self):
        env_name = 'CartPole-v0'
        gym.make(env_name)

    def test_instanceid(self):
        env_name = 'CartPole-v0'
        name = easyagents.easyenv.register(env_name)
        old_count=0
        if name in easyagents.easyenv.EasyEnv._instance_counts:
            old_count = easyagents.easyenv.EasyEnv._instance_counts[name]
        gym.make(name)
        new_count = easyagents.easyenv.EasyEnv._instance_counts[name]
        assert new_count == (old_count + 1)

    def test_LoggingVerbose(self):
        ppo_agent = easyagents.tfagents.PpoAgent('CartPole-v0', training=TrainingFast(), logging=LoggingVerbose())
        ppo_agent.train()
        return

    def test_register_once(self):
        env_name = 'CartPole-v0'
        name = easyagents.easyenv.register(env_name)
        assert name == "Easy_" + env_name
        env = gym.make(name)
        assert env is not None

    def test_register_twice(self):
        env_name = 'CartPole-v0'
        easyagents.easyenv.register(env_name)
        easyagents.easyenv.register(env_name)

    def test_register_properties_set(self):
        env_name = 'CartPole-v0'
        spec = gym.envs.registration.spec(env_name)
        easyenv_name = easyagents.easyenv.register(env_name)
        easy_spec = gym.envs.registration.spec(easyenv_name)

        assert spec.max_episode_seconds == easy_spec.max_episode_seconds
        assert spec.max_episode_steps == easy_spec.max_episode_steps
        assert spec.reward_threshold == easy_spec.reward_threshold

    def test_register_tfagents_suitegymload(self):
        env_name = 'CartPole-v0'
        easyenv_name = easyagents.easyenv.register(env_name)
        tf_env = suite_gym.load(easyenv_name)
        assert tf_env is not None

    def test_set_step_callback(self):
        ppo_agent = easyagents.tfagents.PpoAgent('CartPole-v0', training=TrainingSingleEpisode(),
                                                 logging=LoggingVerbose())
        ppo_agent.train()

        TestEasyEnv.step_callback_call_count = 0
        (reward, steps) = ppo_agent.play_episode(callback=step_callback)
        assert reward > 0
        assert steps > 0
        assert TestEasyEnv.step_callback_call_count > 0


def step_callback(gym_env, action, state, reward, step, done, info):
    TestEasyEnv.step_callback_call_count += 1
    return


if __name__ == '__main__':
    unittest.main()
