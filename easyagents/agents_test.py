import pytest
import unittest
import easyagents.agents
import easyagents.callbacks.log

from easyagents.backends.default import BackendAgentFactory
from easyagents.callbacks.duration import SingleEpisode

_env_name = 'CartPole-v0'


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


class TfAgentsPpoAgentTest(unittest.TestCase):

    def test_callback_single(self):
        agent = easyagents.PpoAgent("CartPole-v0")
        agent.train(SingleEpisode())

    def test_train(self):
        ppo = easyagents.agents.PpoAgent(gym_env_name=_env_name, backend_name='tfagents')
        count = easyagents.callbacks.log.CountCallbacks()
        ppo.train([easyagents.callbacks.log.LogCallbacks(), count, SingleEpisode()])


if __name__ == '__main__':
    unittest.main()
