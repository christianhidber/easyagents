import pytest
import unittest
import easyagents.agents
from easyagents.agents import EasyAgent
from easyagents.core import ModelConfig
from easyagents.backends.default import DefaultBackend

_env_name = 'CartPole-v0'


class BackendRegistrationTest(unittest.TestCase):

    def setup_method(self, method):
        self._oldbackends = easyagents.agents._backends.copy()

    def teardown_method(self, method):
        easyagents.agents._backends = self._oldbackends

    def test_getbackends(self):
        assert easyagents.agents._backends is not None
        assert easyagents.agents.get_backends() is not None

    def test_register(self):
        assert "MyBackend" not in easyagents.agents.get_backends()
        easyagents.agents.register_backend("MyBackend", DefaultBackend())
        assert "MyBackend" in easyagents.agents.get_backends()

    def test_register_backend_empty(self):
        with pytest.raises(AssertionError):
            easyagents.agents.register_backend(backend_name="", backend=DefaultBackend())

    def test_register_backend_nameNone_exception(self):
        with pytest.raises(AssertionError):
            easyagents.agents.register_backend(backend_name=None, backend=DefaultBackend())

    def test_register_backend_backendNone_exception(self):
        with pytest.raises(AssertionError):
            easyagents.agents.register_backend(backend_name="testBackend", backend=None)


class AgentConfigTest(unittest.TestCase):

    def test_create(self):
        assert ModelConfig(gym_env_name=_env_name) is not None
        assert ModelConfig(gym_env_name=_env_name, fc_layers=(10, 20)) is not None

    def test_create_envNotRegistered_exception(self):
        with pytest.raises(AssertionError):
            ModelConfig(gym_env_name="MyEnv-v0")

    def test_create_envNotnameNotSet_exception(self):
        with pytest.raises(AssertionError):
            ModelConfig(gym_env_name=None)

    def test_create_fclayersEmpty_exception(self):
        with pytest.raises(AssertionError):
            ModelConfig(gym_env_name=_env_name, fc_layers=())

    def test_create_fclayersSimpleInt(self):
        assert ModelConfig(gym_env_name=_env_name, fc_layers=10) is not None

    def test_create_fclayersNegativeValue(self):
        with pytest.raises(AssertionError):
            ModelConfig(gym_env_name=_env_name, fc_layers=-10)


class EasyAgentsTest(unittest.TestCase):
    class SimpleAgent(EasyAgent):
        def __init__(self, gym_env_name: str, fc_layers=None):
            super().__init__()
            self._agent_config = ModelConfig(gym_env_name=gym_env_name, fc_layers=fc_layers)
            return

    def test_create(self):
        assert EasyAgentsTest.SimpleAgent(gym_env_name=_env_name) is not None
        assert EasyAgentsTest.SimpleAgent(gym_env_name=_env_name, fc_layers=(10, 20)) is not None

    def test_create_envNotRegistered_exception(self):
        with pytest.raises(AssertionError):
            EasyAgentsTest.SimpleAgent(gym_env_name="MyEnv-v0")

    def test_create_envNotnameNotSet_exception(self):
        with pytest.raises(AssertionError):
            EasyAgentsTest.SimpleAgent(gym_env_name=None)

    def test_create_fclayersEmpty_exception(self):
        with pytest.raises(AssertionError):
            EasyAgentsTest.SimpleAgent(gym_env_name=_env_name, fc_layers=())

    def test_create_fclayersSimpleInt(self):
        assert EasyAgentsTest.SimpleAgent(gym_env_name=_env_name, fc_layers=10) is not None

    def test_create_fclayersNegativeValue(self):
        with pytest.raises(AssertionError):
            EasyAgentsTest.SimpleAgent(gym_env_name=_env_name, fc_layers=-10) is not None


if __name__ == '__main__':
    unittest.main()
