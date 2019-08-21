import unittest

import easyagents.core as core
import easyagents.backends.core as bcore
import easyagents.traincallbacks.debug as debug


class BackendAgentTest(unittest.TestCase):
    class DebugAgent(bcore.BackendAgent):
        def __init__(self):
            self.train_count = 0

        def train_implementation(self, train_context: core.TrainContext):
            self.train_count += 1
            for i in range(train_context.num_iterations):
                self.on_iteration_begin()
                self.on_iteration_end(0.123 + i)

    def test_train_emptyArgs(self):
        agent = BackendAgentTest.DebugAgent()
        agent.train(train_context=core.SingleEpisodeTrainContext(),
                    train_callbacks=[], play_callbacks=[], api_callbacks=[])

    def test_train_missingArgs(self):
        agent = BackendAgentTest.DebugAgent()
        context=core.SingleEpisodeTrainContext()
        with self.assertRaises(AssertionError):
            agent.train(train_context=None,
                        train_callbacks=[], play_callbacks=[], api_callbacks=[])
        with self.assertRaises(AssertionError):
            agent.train(train_context=context,
                        train_callbacks=None, play_callbacks=[], api_callbacks=[])
        with self.assertRaises(AssertionError):
            agent.train(train_context=context,
                        train_callbacks=[], play_callbacks=None, api_callbacks=[])
        with self.assertRaises(AssertionError):
            agent.train(train_context=context,
                        train_callbacks=[], play_callbacks=[], api_callbacks=None)

    def test_train_callbacks(self):
        agent = BackendAgentTest.DebugAgent()
        callback = debug.Count()
        train_context = core.SingleEpisodeTrainContext()
        train_context.num_iterations=2
        agent.train(train_context=train_context,
                    train_callbacks=[callback], play_callbacks=[], api_callbacks=[])
        assert callback.train_begin_count == 1
        assert callback.train_end_count == 1
        assert callback.iteration_begin_count == 2
        assert callback.iteration_end_count == 2
        assert train_context.current_episode in train_context.loss
