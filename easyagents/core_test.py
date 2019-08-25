import pytest
import unittest

from easyagents.core import ModelConfig

_env_name = 'CartPole-v0'

class ModelConfigTest(unittest.TestCase):

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


if __name__ == '__main__':
    unittest.main()
