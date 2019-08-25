import unittest
import logging

import easyagents.core as core
import easyagents.backends.noop as noop
import easyagents.callbacks.debug


class BackendAgentTest(unittest.TestCase):
    class DebugAgent(noop.BackendAgent):
        def __init__(self):
            super().__init__(core.ModelConfig(gym_env_name='CartPole-v0'), action=1)

    def test_log(self):
        agent = BackendAgentTest.DebugAgent()
        log = easyagents.callbacks.debug.Log()
        agent.train(train_context=core.SingleEpisodeTrainContext(),callbacks=[log])
