"""This module contains the backend for a constant action agent implementation.

    All agents created by the debug backend execute a "no operations" train loop,
    without an underlying model / neural network.
"""

from typing import Tuple, Dict, Type
from easyagents import core
from easyagents.backends import core as bcore
import easyagents.agents
import gym


class DebugAgentFactory(bcore.BackendAgentFactory):
    backend_name: str = 'debug'

    def get_algorithms(self) -> Dict[Type, Type[bcore._BackendAgent]]:
        """Yields a mapping of EasyAgent types to the implementations provided by this backend."""
        return {easyagents.agents.DqnAgent: DebugAgent,
                easyagents.agents.PpoAgent: DebugAgent,
                easyagents.agents.RandomAgent: DebugAgent,
                easyagents.agents.ReinforceAgent: DebugAgent}


class DebugAgent(bcore._BackendAgent):

    def __init__(self, model_config: core.ModelConfig, action=None):
        """Simple constant action agent.

        Args:
            model_config: containing the gym_env_name to "train" on
            action: the action to take in all steps or None. If None no steps are taken.
        """
        super().__init__(model_config, backend_name=DebugAgentFactory.backend_name, tf_eager_execution=True)
        self.action = action

    def play_implementation(self, play_context: core.PlayContext):
        env = gym.make(self.model_config.gym_env_name)
        while True:
            self.on_play_episode_begin(env=env)
            env.reset()
            if self.action is not None:
                done = False
                while not done:
                    (observation, reward, done, info) = env.step(self.action)
            self.on_play_episode_end()
            if play_context.play_done:
                break

    def train_implementation(self, train_context: core.TrainContext):
        assert isinstance(train_context, core.EpisodesTrainContext)
        tc: core.EpisodesTrainContext = train_context

        env = gym.make(self.model_config.gym_env_name)
        for i in range(tc.num_iterations):
            self.on_train_iteration_begin()
            for e in range(tc.num_episodes_per_iteration):
                env.reset()
                if self.action is not None:
                    done = False
                    while not done:
                        (observation, reward, done, info) = env.step(self.action)
            self.on_train_iteration_end(0)
            if tc.training_done:
                break

    def load_implementation(self, directory: str):
        pass

    def save_implementation(self, directory: str):
        pass


class InvariantCallback(core.AgentCallback):
    """Validates the callback invariants"""

    def on_play_begin(self, agent_context: core.AgentContext):
        assert agent_context.play.gym_env is None

    def on_play_end(self, agent_context: core.AgentContext):
        assert agent_context.play.gym_env is not None

    def on_play_episode_begin(self, agent_context: core.AgentContext):
        assert agent_context.play.gym_env is not None

    def on_play_episode_end(self, agent_context: core.AgentContext):
        assert agent_context.play.gym_env is not None

    def on_play_step_begin(self, agent_context: core.AgentContext, action):
        assert agent_context.play.gym_env is not None

    def on_play_step_end(self, agent_context: core.AgentContext, action, step_result: Tuple):
        assert agent_context.play.gym_env is not None
