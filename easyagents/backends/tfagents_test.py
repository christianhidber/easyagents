import unittest

from easyagents import core
from easyagents.backends import tfagents


class TfPpoAgentTest(unittest.TestCase):

    def test_train(self):
        model_config = core.ModelConfig("CartPole-v0")
        train_context = core.SingleEpisodeTrainContext()
        ppoAgent = tfagents.TfPpoAgent(model_config=model_config )
        ppoAgent.train(train_context=train_context,callbacks=[])


if __name__ == '__main__':
    unittest.main()
