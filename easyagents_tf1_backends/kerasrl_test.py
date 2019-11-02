import unittest
import pytest

import easyagents
import easyagents.backends.core
import easyagents.core as core
from  easyagents.backends import kerasrl
from easyagents.callbacks import duration,log

easyagents.agents.seed = 0

class KerasRlTest(unittest.TestCase):

    @pytest.mark.skipif(easyagents.backends.core._tensorflow_v2_eager_enabled, reason="tfv2 active")
    @pytest.mark.tfv1
    def test_cem(self):
        easyagents.agents.seed = 0
        model_config = core.ModelConfig("CartPole-v0", fc_layers=(100,))
        tc = core.CemTrainContext()
        tc.num_iterations=100
        tc.num_episodes_per_iteration=50
        tc.max_steps_per_episode=200
        tc.elite_set_fraction=0.1
        tc.num_steps_buffer_preload=2000
        cemAgent = kerasrl.KerasRlCemAgent(model_config=model_config)
        cemAgent.train(train_context=tc,callbacks=[log.Agent(), log.Iteration()])


    @pytest.mark.skipif(easyagents.backends.core._tensorflow_v2_eager_enabled, reason="tfv2 active")
    @pytest.mark.tfv1
    def test_dqn(self):
        easyagents.agents.seed = 0
        model_config = core.ModelConfig("CartPole-v0")
        tc = core.StepsTrainContext()
        dqnAgent = kerasrl.KerasRlDqnAgent(model_config=model_config)
        dqnAgent.train(train_context=tc,
                       callbacks=[duration.Fast(), log.Agent(), log.Step(), log.Iteration()])