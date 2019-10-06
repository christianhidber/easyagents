import unittest
import pytest

import easyagents
import easyagents.backends.core
import easyagents.core as core
from  easyagents.backends import kerasrl
from easyagents.callbacks import duration,log, plot

class KerasRlTest(unittest.TestCase):

    @pytest.mark.skipif(easyagents.backends.core._tensorflow_v2_eager_enabled)
    def test_train_tfv1(self):
        easyagents.agents.seed = 0
        model_config = core.ModelConfig("CartPole-v0")
        tc = core.DqnTrainContext()
        dqnAgent = kerasrl.KerasRlDqnAgent(model_config=model_config)
        dqnAgent.train(train_context=tc,
                       callbacks=[duration._SingleEpisode(), log.Agent(), log.Step(), log.Iteration()])