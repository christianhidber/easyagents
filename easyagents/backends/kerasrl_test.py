import unittest

import easyagents
import easyagents.core as core
from  easyagents.backends import kerasrl
from easyagents.callbacks import duration,log

class KerasRlTest(unittest.TestCase):

    def test_train(self):
        easyagents.agents.seed = 0
        model_config = core.ModelConfig("CartPole-v0")
        tc = core.DqnTrainContext()
        dqnAgent = kerasrl.KerasRlDqnAgent(model_config=model_config)
        dqnAgent.train(train_context=tc,
                       callbacks=[duration._SingleEpisode(), log.Agent(), log.Step(), log.Iteration()])