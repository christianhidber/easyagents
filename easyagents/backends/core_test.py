import unittest

import easyagents.core as core
import easyagents.backends.noop as noop
import easyagents.traincallbacks.debug
import easyagents.apicallbacks.debug


class BackendAgentTest(unittest.TestCase):
    class DebugAgent(noop.BackendAgent):
        def __init__(self):
            super().__init__(core.ModelConfig(gym_env_name='CartPole-v0'), action=1)

    def test_train_emptyArgs(self):
        agent = BackendAgentTest.DebugAgent()
        train_context = core.SingleEpisodeTrainContext()
        agent.train(train_context=train_context, train_callbacks=[], play_callbacks=[], api_callbacks=[])

    def test_train_missingArgs(self):
        agent = BackendAgentTest.DebugAgent()
        context = core.SingleEpisodeTrainContext()
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
        traincount = easyagents.traincallbacks.debug.Count()
        train_context = core.SingleEpisodeTrainContext()
        train_context.num_iterations = 2
        train_context.seed = 0
        apicount = easyagents.apicallbacks.debug.Count()
        agent.train(train_context=train_context,
                    train_callbacks=[traincount],
                    play_callbacks=[],
                    api_callbacks=[apicount])
        assert traincount.train_begin_count == traincount.train_end_count == 1
        assert traincount.iteration_begin_count == traincount.iteration_end_count == 2
        assert train_context.episodes_done_in_training == 2
        assert train_context.episodes_done_in_iteration == 1
        assert train_context.episodes_done_in_training in train_context.loss
        assert train_context.steps_done_in_training > train_context.steps_done_in_iteration > 0
        assert apicount.gym_init_begin_count == apicount.gym_init_end_count > 0
        assert apicount.gym_reset_begin_count == apicount.gym_reset_end_count > 0
        assert apicount.gym_step_begin_count == apicount.gym_step_end_count > 0
