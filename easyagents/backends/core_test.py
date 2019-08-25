import unittest

import easyagents.core as core
import easyagents.backends.noop as noop
import easyagents.callbacks.debug


class BackendAgentTest(unittest.TestCase):
    class DebugAgent(noop.BackendAgent):
        def __init__(self):
            super().__init__(core.ModelConfig(gym_env_name='CartPole-v0'), action=1)

    def test_train_emptyArgs(self):
        agent = BackendAgentTest.DebugAgent()
        train_context = core.SingleEpisodeTrainContext()
        agent.train(train_context=train_context, callbacks=[])

    def test_train_missingArgs(self):
        agent = BackendAgentTest.DebugAgent()
        context = core.SingleEpisodeTrainContext()
        with self.assertRaises(AssertionError):
            agent.train(train_context=None, callbacks=[])
        with self.assertRaises(AssertionError):
            agent.train(train_context=context, callbacks=None)

    def test_train_callbacks(self):
        agent = BackendAgentTest.DebugAgent()
        count = easyagents.callbacks.debug.Count()
        train_context = core.SingleEpisodeTrainContext()
        train_context.num_iterations = 2
        train_context.seed = 0
        agent.train(train_context=train_context,callbacks=[count])
        assert count.train_begin_count == count.train_end_count == 1
        assert count.train_iteration_begin_count == count.train_iteration_end_count == 2
        assert train_context.episodes_done_in_training == 2
        assert train_context.episodes_done_in_iteration == 1
        assert train_context.episodes_done_in_training in train_context.loss
        assert train_context.steps_done_in_training > train_context.steps_done_in_iteration > 0
        assert count.gym_init_begin_count == count.gym_init_end_count > 0
        assert count.gym_reset_begin_count == count.gym_reset_end_count > 0
        assert count.gym_step_begin_count == count.gym_step_end_count > 0
