import unittest

from easyagents.config import LoggingMinimal
from easyagents.config import LoggingNormal
from easyagents.config import TrainingFast
from easyagents.config import TrainingSingleEpisode
from easyagents.easyenv import EasyEnv
from easyagents.tfagents import DqnAgent
from easyagents.tfagents import PpoAgent
from easyagents.tfagents import ReinforceAgent
from easyagents.tfagents import VpgAgent


class TestTfAgents(unittest.TestCase):
    count = 0

    def setUp(self):
        self.ppo = PpoAgent('CartPole-v0', training=TrainingSingleEpisode(), logging=LoggingMinimal())

    def test_get_easyenv(self):
        tfenv = self.ppo._create_tfagent_env()
        easyenv = self.ppo._get_easyenv(tfenv)
        assert isinstance(easyenv, EasyEnv)

    def test_play_episode_nocallback(self):
        self.ppo.train()
        reward, steps = self.ppo.play_episode()
        assert isinstance(reward, float)
        assert isinstance(steps, int)
        assert steps > 0

    def test_play_episode_withcallback(self):
        self.ppo.train()
        TestTfAgents.count = 0
        reward, steps = self.ppo.play_episode(
            callback=lambda gym_env, action, state, reward, step, done, info:
            TestTfAgents.increment_count())
        assert steps == TestTfAgents.count

    def test_play_episode_withmaxsteps(self):
        self.ppo.train()
        reward, steps = self.ppo.play_episode(max_steps=2)
        assert steps == 2

    @staticmethod
    def increment_count():
        TestTfAgents.count += 1


class TestDqnAgent(unittest.TestCase):

    def setUp(self):
        self.gym_env_name = 'CartPole-v0'

    def test_create(self):
        dqn_agent = DqnAgent(self.gym_env_name)
        self.assertIsNotNone(dqn_agent, "failed to create a tfagents.DqnAgent instance for " + self.gym_env_name)

    def test_train(self):
        dqn_agent = DqnAgent(self.gym_env_name, training=TrainingFast(), logging=LoggingNormal())
        dqn_agent.train()


class TestPpoAgent(unittest.TestCase):

    def setUp(self):
        self.gym_env_name = 'CartPole-v0'

    def test_create(self):
        ppo_agent = PpoAgent(self.gym_env_name)
        self.assertIsNotNone(ppo_agent, "failed to create a tfagents.PpoAgent instance for " + self.gym_env_name)

    def test_train(self):
        ppo_agent = PpoAgent(self.gym_env_name, training=TrainingFast())
        ppo_agent.train()

    def test_str(self):
        ppo_agent = PpoAgent(self.gym_env_name, training=TrainingFast())
        result = str(ppo_agent)
        print(result)


class TestReinforceAgent(unittest.TestCase):

    def setUp(self):
        self.gym_env_name = 'CartPole-v0'

    def test_create(self):
        agent = ReinforceAgent(self.gym_env_name)
        self.assertIsNotNone(agent, "failed to create a tfagents.ReinforceAgent instance for " + self.gym_env_name)

    def test_create_with_alias(self):
        agent = VpgAgent(self.gym_env_name)
        self.assertIsNotNone(agent, "failed to create a tfagents.ReinforceAgent instance for " + self.gym_env_name)

    def test_train(self):
        agent = ReinforceAgent(self.gym_env_name, training=TrainingFast())
        agent.train()


if __name__ == '__main__':
    unittest.main()
