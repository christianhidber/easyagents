import unittest

import pytest

import easyagents.backends.core
import easyagents.backends.debug
from easyagents.agents import PpoAgent
from easyagents.callbacks.duration import Fast
from easyagents.core import ModelConfig, AgentCallback, AgentContext, PpoTrainContext, PlayContext

_stepcount_name = easyagents.env._StepCountEnv.register_with_gym()


class AgentContextTest(unittest.TestCase):
    class PlayCallback(AgentCallback):
        def __init__(self):
            self.play_called = False
            self.train_called = False

        def on_play_begin(self, ac: AgentContext):
            assert ac.is_play
            assert not ac.is_eval
            assert not ac.is_train
            self.play_called = True

        def on_train_begin(self, agent_context: AgentContext):
            self.train_called = True

    class TrainCallback(AgentCallback):

        def __init__(self):
            self.play_called = False
            self.train_called = False

        def on_play_begin(self, agent_context: AgentContext):
            assert not agent_context.is_play
            assert agent_context.is_eval
            assert not agent_context.is_train
            self.play_called = True

        def on_train_begin(self, agent_context: AgentContext):
            assert not agent_context.is_play
            assert not agent_context.is_eval
            assert agent_context.is_train
            self.train_called = True

    def test_agentcontext_train(self):
        b = easyagents.backends.debug.DebugAgentFactory()
        a = b.create_agent(PpoAgent, ModelConfig(_stepcount_name))
        c = AgentContextTest.TrainCallback()
        a.train(callbacks=[Fast(), c], train_context=PpoTrainContext())
        assert c.train_called
        assert c.play_called

    def test_agentcontext_play(self):
        b = easyagents.backends.debug.DebugAgentFactory()
        a = b.create_agent(PpoAgent, ModelConfig(_stepcount_name))
        c = AgentContextTest.PlayCallback()
        pc = PlayContext()
        pc.num_episodes = 10
        pc.max_steps_per_episode = 10
        a.play(callbacks=[Fast(), c], play_context=pc)
        assert not c.train_called
        assert c.play_called


class ModelConfigTest(unittest.TestCase):

    def test_create(self):
        assert ModelConfig(gym_env_name=_stepcount_name) is not None
        assert ModelConfig(gym_env_name=_stepcount_name, fc_layers=(10, 20)) is not None

    def test_create_envNotRegistered_exception(self):
        with pytest.raises(AssertionError):
            ModelConfig(gym_env_name="MyEnv-v0")

    def test_create_envNotnameNotSet_exception(self):
        with pytest.raises(AssertionError):
            # noinspection PyTypeChecker
            ModelConfig(gym_env_name=None)

    def test_create_fclayersEmpty_exception(self):
        with pytest.raises(AssertionError):
            ModelConfig(gym_env_name=_stepcount_name, fc_layers=())

    def test_create_fclayersSimpleInt(self):
        assert ModelConfig(gym_env_name=_stepcount_name, fc_layers=10) is not None

    def test_create_fclayersNegativeValue(self):
        with pytest.raises(AssertionError):
            ModelConfig(gym_env_name=_stepcount_name, fc_layers=-10)

    def test_to_from_dict(self):
        mc = ModelConfig(gym_env_name=_stepcount_name, fc_layers=10, seed=123)
        mc_dict = mc._to_dict()
        mc2 = ModelConfig._from_dict(mc_dict)
        assert mc2.fc_layers == mc.fc_layers
        assert mc2.original_env_name == mc.original_env_name
        assert mc2.seed == mc.seed


if __name__ == '__main__':
    unittest.main()
