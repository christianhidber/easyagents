import unittest

from easyagents import core, env
from easyagents.backends import hk
from easyagents.callbacks import duration, log


class HuskarlDqnAgentTest(unittest.TestCase):

    def setUp(self):
        self.env_name = env._StepCountEnv.register_with_gym()

    def test_train(self):
        model_config = core.ModelConfig("CartPole-v0")
        tc = core.DqnTrainContext()
        dqnAgent = hk.HuskarlDqnAgent(model_config=model_config)
        dqnAgent.train(train_context=tc, callbacks=[log.Step(), duration.Fast()])