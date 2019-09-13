import pytest
import unittest

from easyagents import env, core, agents
from easyagents.callbacks import duration, log, plot
from easyagents.backends import default

_env_name = env._StepCountEnv.register_with_gym()


# noinspection PyTypeChecker
class BackendRegistrationTest(unittest.TestCase):

    def setUp(self):
        self._oldbackends = agents._backends.copy()

    def tearDown(self):
        agents._backends = self._oldbackends

    def test_getbackends(self):
        assert agents._backends is not None
        assert agents.get_backends() is not None

    def test_prepare_callbacks(self):
        agent = agents.PpoAgent("CartPole-v0")
        c = [plot.ToMovie(), plot.Rewards()]
        d = agent._prepare_callbacks(c)
        assert isinstance(d[0], plot._PreProcess)
        assert isinstance(d[1], plot.Rewards)
        assert isinstance(d[-2], plot._PostProcess)
        assert isinstance(d[-1], plot.ToMovie)

    def test_register(self):
        assert "MyBackend" not in agents.get_backends()
        agents.register_backend("MyBackend", default.BackendAgentFactory())
        assert "MyBackend" in agents.get_backends()

    def test_register_backend_empty(self):
        with pytest.raises(AssertionError):
            agents.register_backend(backend_name="", backend=default.BackendAgentFactory())

    def test_register_backend_nameNone_exception(self):
        with pytest.raises(AssertionError):
            agents.register_backend(backend_name=None, backend=default.BackendAgentFactory())

    def test_register_backend_backendNone_exception(self):
        with pytest.raises(AssertionError):
            agents.register_backend(backend_name="testBackend", backend=None)


class TfAgentsDqnAgentTest(unittest.TestCase):

    def test_train(self):
        dqnAgent = agents.DqnAgent('CartPole-v0', fc_layers=(100,))
        tc: core.TrainContext = dqnAgent.train([log.Duration(), log.Iteration()],
                                               num_iterations=10000,
                                               num_iterations_between_log=200,
                                               num_iterations_between_eval=1000,
                                               max_steps_per_episode=200,
                                               default_plots=False)
        (min_steps, avg_steps, max_steps) = tc.eval_steps[tc.episodes_done_in_training]
        assert avg_steps >= 150
        assert max_steps == 200


# noinspection PyTypeChecker
class TfAgentsPpoAgentTest(unittest.TestCase):

    def test_callback_single(self):
        env._StepCountEnv.clear()
        agent = agents.PpoAgent(_env_name)
        agent.train(duration._SingleEpisode())
        assert env._StepCountEnv.reset_count <= 2

    def test_train_cartpole(self):
        ppo = agents.PpoAgent(gym_env_name="CartPole-v0", backend='tfagents')
        tc = core.ActorCriticTrainContext()
        tc.num_iterations = 3
        tc.num_episodes_per_iteration = 10
        tc.max_steps_per_episode = 500
        tc.num_epochs_per_iteration = 5
        tc.num_iterations_between_eval = 2
        tc.num_episodes_per_eval = 5
        ppo.train([log.Iteration()], train_context=tc)

    def test_train_single_episode(self):
        ppo = agents.PpoAgent(gym_env_name=_env_name, backend='tfagents')
        count = log._CallbackCounts()
        ppo.train([log.Agent(), count, duration._SingleEpisode()])
        assert count.gym_init_begin_count == count.gym_init_end_count == 1
        assert count.gym_step_begin_count == count.gym_step_end_count <= 10

    def test_play_single_episode(self):
        ppo = agents.PpoAgent(gym_env_name=_env_name, backend='tfagents')
        count = log._CallbackCounts()
        cb = [log.Agent(), count, duration._SingleEpisode()]
        ppo.train(duration._SingleEpisode())
        ppo.play(cb)
        assert count.gym_init_begin_count == count.gym_init_end_count == 1
        assert count.gym_step_begin_count == count.gym_step_end_count <= 10


if __name__ == '__main__':
    unittest.main()
