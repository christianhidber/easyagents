import unittest

import easyagents.core as core
import easyagents.env
import easyagents.backends.debug as debug
import easyagents.callbacks.log


# noinspection PyTypeChecker
class BackendAgentTest(unittest.TestCase):
    env_name = easyagents.env._StepCountEnv.register_with_gym()

    def setUp(self):
        self.tc = core.TrainContext()
        self.tc.num_episodes_per_iteration = 1
        self.tc.num_iterations = 1
        self.tc.num_episodes_per_eval=2
        self.tc.max_steps_per_episode = 5
        self.pc = core.PlayContext(self.tc)

    class DebugAgent(debug.BackendAgent):
        def __init__(self):
            super().__init__(core.ModelConfig(gym_env_name=BackendAgentTest.env_name), action=1)

    def test_play_playcontext(self):
        agent = BackendAgentTest.DebugAgent()
        count = easyagents.callbacks.log.CountCallbacks()
        agent.play(play_context=self.pc,callbacks=[count])
        assert self.pc.play_done == True
        assert self.pc.episodes_done == self.pc.num_episodes
        assert self.pc.steps_done == self.pc.num_episodes * self.pc.max_steps_per_episode == 10
        assert self.pc.steps_done_in_episode == self.pc.max_steps_per_episode == 5


    def test_train_emptyArgs(self):
        agent = BackendAgentTest.DebugAgent()
        agent.train(train_context=self.tc, callbacks=[])
        assert self.tc.training_done
        assert self.tc.iterations_done_in_training == 1
        assert self.tc.episodes_done_in_iteration == 1
        assert self.tc.episodes_done_in_training == 1

    def test_train_missingArgs(self):
        agent = BackendAgentTest.DebugAgent()
        with self.assertRaises(AssertionError):
            agent.train(train_context=None, callbacks=[])
        with self.assertRaises(AssertionError):
            agent.train(train_context=self.tc, callbacks=None)

    def test_train_traincontext(self):
        agent = BackendAgentTest.DebugAgent()
        count = easyagents.callbacks.log.CountCallbacks()
        tc = self.tc
        tc.num_iterations = 3
        tc.num_episodes_per_iteration=2
        tc.max_steps_per_episode=10
        tc.num_episodes_per_eval=5
        tc.num_iterations_between_eval=2
        agent.train(train_context=tc, callbacks=[count])
        assert count.train_begin_count == count.train_end_count == 1
        assert count.train_iteration_begin_count == count.train_iteration_end_count == 3
        assert tc.episodes_done_in_training == 6
        assert tc.episodes_done_in_iteration == 2
        assert tc.episodes_done_in_training in tc.loss
        assert tc.steps_done_in_training == tc.num_iterations * tc.num_episodes_per_iteration * \
               tc.max_steps_per_episode
        assert tc.steps_done_in_iteration == tc.num_episodes_per_iteration * tc.max_steps_per_episode

        assert 0 in tc.eval_rewards
        assert 0 in tc.eval_steps
        assert tc.num_iterations_between_eval * tc.num_episodes_per_iteration in tc.eval_rewards
        assert tc.num_iterations_between_eval * tc.num_episodes_per_iteration in tc.eval_steps
        assert tc.episodes_done_in_training in tc.eval_rewards
        assert tc.episodes_done_in_training in tc.eval_steps

        assert count.gym_init_begin_count == count.gym_init_end_count > 0
        assert count.gym_reset_begin_count == count.gym_reset_end_count > 0
        assert count.gym_step_begin_count == count.gym_step_end_count > 0
