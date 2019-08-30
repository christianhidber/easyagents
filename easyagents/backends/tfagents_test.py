import unittest

from easyagents import core
from easyagents.backends import tfagents
import easyagents.env


class TfPpoAgentTest(unittest.TestCase):

    def setUp(self):
        self.env_name = easyagents.env._StepCountEnv.register_with_gym()

    def test_train(self):
        model_config = core.ModelConfig(self.env_name)
        train_context = core.TrainContext()
        train_context.num_iterations = 1
        train_context.num_episodes_per_iteration = 1
        ppoAgent = tfagents.TfPpoAgent(model_config=model_config)
        ppoAgent.train(train_context=train_context, callbacks=[])


if __name__ == '__main__':
    unittest.main()
