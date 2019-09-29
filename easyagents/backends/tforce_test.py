import unittest

from easyagents import core, env
from easyagents.backends import tforce
from easyagents.callbacks import duration, log


class TensorforceAgentTest(unittest.TestCase):

    def setUp(self):
        self.env_name = env._StepCountEnv.register_with_gym()

    def test_ppo_train(self):
        model_config = core.ModelConfig("CartPole-v0")
        tc = core.ActorCriticTrainContext()
        tc.num_iterations=11
        tc.num_episodes_per_iteration=5
        ppoAgent = tforce.TforcePpoAgent(model_config=model_config)
        ppoAgent.train(train_context=tc, callbacks=[log.Iteration(), log.Agent()])

    def test_reinforce_train(self):
        model_config = core.ModelConfig("CartPole-v0")
        tc = core.EpisodesTrainContext()
        tc.num_iterations=11
        tc.num_episodes_per_iteration=5
        randomAgent = tforce.TforceReinforceAgent(model_config=model_config)
        randomAgent.train(train_context=tc, callbacks=[log.Iteration(), log.Agent(), log.Step()])


if __name__ == '__main__':
    unittest.main()
