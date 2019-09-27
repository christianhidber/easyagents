import unittest

from easyagents import core, env
from easyagents.backends import tensorforce
from easyagents.callbacks import duration, log


class TensorforcePpoAgentTest(unittest.TestCase):

    def setUp(self):
        self.env_name = env._StepCountEnv.register_with_gym()

    def test_train(self):
        model_config = core.ModelConfig("CartPole-v0")
        tc = core.ActorCriticTrainContext()
        ppoAgent = tensorforce.TensorforcePpoAgent(model_config=model_config)
        ppoAgent.train(train_context=tc, callbacks=[duration.Fast(), log.Iteration()])


if __name__ == '__main__':
    unittest.main()
