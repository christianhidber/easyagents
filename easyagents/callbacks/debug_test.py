import unittest
import logging

import easyagents.core as core
import easyagents.backends.debug as debug
import easyagents.callbacks.debug


class BackendAgentTest(unittest.TestCase):
    class DebugAgent(debug.BackendAgent):
        def __init__(self):
            super().__init__(core.ModelConfig(gym_env_name='CartPole-v0'), action=1)

    def test_log(self):
        agent = BackendAgentTest.DebugAgent()
        log = easyagents.callbacks.debug.Log()
        train_context = core.TrainContext()
        train_context.num_iterations=1
        train_context.num_episodes_per_iteration=1
        agent.train(train_context=train_context,callbacks=[log])
