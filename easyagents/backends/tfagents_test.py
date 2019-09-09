import unittest

from easyagents import core
from easyagents.backends import tfagents
import easyagents.env
from easyagents.callbacks.duration import _SingleEpisode, Fast
from easyagents.callbacks.log import Step, Agent, _Callbacks, Iteration


class TfDqnAgentTest(unittest.TestCase):

    def setUp(self):
        self.env_name = easyagents.env._StepCountEnv.register_with_gym()

    def test_train(self):
        model_config = core.ModelConfig("CartPole-v0")
        tc = core.DqnTrainContext()
        dqnAgent = tfagents.TfDqnAgent(model_config=model_config)
        dqnAgent.train(train_context=tc, callbacks=[_SingleEpisode(), Iteration()])


class TfPpoAgentTest(unittest.TestCase):

    def setUp(self):
        self.env_name = easyagents.env._StepCountEnv.register_with_gym()

    def test_train(self):
        model_config = core.ModelConfig("CartPole-v0")
        tc = core.ActorCriticTrainContext()
        ppoAgent = tfagents.TfPpoAgent(model_config=model_config)
        ppoAgent.train(train_context=tc, callbacks=[Fast(), Iteration()])


if __name__ == '__main__':
    unittest.main()
