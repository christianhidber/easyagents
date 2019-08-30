import pytest
import unittest
import easyagents.agents
import easyagents.callbacks.log
import easyagents.env
import easyagents.core

from easyagents.backends.default import BackendAgentFactory
from easyagents.callbacks.duration import SingleEpisode, Fast
from easyagents.callbacks.log import LogAgent, LogIteration, LogStep

_env_name = easyagents.env._StepCountEnv.register_with_gym()


# noinspection PyTypeChecker
class BackendRegistrationTest(unittest.TestCase):

    def setUp(self):
        self._oldbackends = easyagents.agents._backends.copy()

    def tearDown(self):
        easyagents.agents._backends = self._oldbackends

    def test_getbackends(self):
        assert easyagents.agents._backends is not None
        assert easyagents.agents.get_backends() is not None

    def test_register(self):
        assert "MyBackend" not in easyagents.agents.get_backends()
        easyagents.agents.register_backend("MyBackend", BackendAgentFactory())
        assert "MyBackend" in easyagents.agents.get_backends()

    def test_register_backend_empty(self):
        with pytest.raises(AssertionError):
            easyagents.agents.register_backend(backend_name="", backend=BackendAgentFactory())

    def test_register_backend_nameNone_exception(self):
        with pytest.raises(AssertionError):
            easyagents.agents.register_backend(backend_name=None, backend=BackendAgentFactory())

    def test_register_backend_backendNone_exception(self):
        with pytest.raises(AssertionError):
            easyagents.agents.register_backend(backend_name="testBackend", backend=None)


# noinspection PyTypeChecker
class TfAgentsPpoAgentTest(unittest.TestCase):

    def test_callback_single(self):
        agent = easyagents.PpoAgent(_env_name)
        agent.train(SingleEpisode())

    def test_train_cartpole(self):
        ppo = easyagents.agents.PpoAgent(gym_env_name="CartPole-v0", backend_name='tfagents')
        tc = easyagents.core.TrainContext()
        tc.num_iterations = 100
        tc.num_episodes_per_iteration = 10
        tc.max_steps_per_episode = 500
        tc.num_epochs_per_iteration = 5
        tc.num_iterations_between_eval = 5
        tc.num_episodes_per_eval = 10
        ppo.train([LogIteration()], train_context=tc)

    def test_train_single_episode(self):
        ppo = easyagents.agents.PpoAgent(gym_env_name=_env_name, backend_name='tfagents')
        count = easyagents.callbacks.log.CountCallbacks()
        ppo.train([easyagents.callbacks.log.LogAgent(), count, SingleEpisode()])
        assert count.gym_init_begin_count == count.gym_init_end_count == 1
        assert count.gym_step_begin_count == count.gym_step_end_count <= 10

    def test_play_single_episode(self):
        ppo = easyagents.agents.PpoAgent(gym_env_name=_env_name, backend_name='tfagents')
        count = easyagents.callbacks.log.CountCallbacks()
        cb = [easyagents.callbacks.log.LogAgent(), count, SingleEpisode()]
        ppo.train(SingleEpisode())
        ppo.play(cb)
        assert count.gym_init_begin_count == count.gym_init_end_count == 1
        assert count.gym_step_begin_count == count.gym_step_end_count <= 10


if __name__ == '__main__':
    unittest.main()
