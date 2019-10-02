import unittest

from easyagents import core, env
from easyagents.backends import tforce
from easyagents.callbacks import duration, log


class TensorforceAgentTest(unittest.TestCase):

    def setUp(self):
        self.env_name = env._StepCountEnv.register_with_gym()

    def test_dqn_train(self):
        model_config = core.ModelConfig("CartPole-v0")
        tc = core.DqnTrainContext()
        dqnAgent = tforce.TforceDqnAgent(model_config=model_config)
        dqnAgent.train(train_context=tc, callbacks=[log.Iteration(), log.Agent(), duration.Fast()])

    def test_ppo_train(self):
        model_config = core.ModelConfig("CartPole-v0")
        tc = core.ActorCriticTrainContext()
        ppoAgent = tforce.TforcePpoAgent(model_config=model_config)
        ppoAgent.train(train_context=tc, callbacks=[log.Iteration(), log.Agent(), duration.Fast()])

    def test_reinforce_train(self):
        model_config = core.ModelConfig("CartPole-v0")
        tc = core.EpisodesTrainContext()
        randomAgent = tforce.TforceReinforceAgent(model_config=model_config)
        randomAgent.train(train_context=tc, callbacks=[log.Iteration(), log.Agent(), duration.Fast()])


if __name__ == '__main__':
    unittest.main()
