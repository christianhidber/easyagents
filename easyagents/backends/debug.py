"""This module contains the backend for a constant action agent implementation.

    All agents created by the debug backend execute a "no operations" train loop,
    without an underlying model / neural network.
"""

from easyagents import core
from easyagents.backends import core as bcore
import gym


class BackendAgentFactory(bcore.BackendAgentFactory):
    name = 'debug'

    def create_dqn_agent(self, model_config: core.ModelConfig) -> bcore._BackendAgent:
        return BackendAgent(model_config=model_config)

    def create_ppo_agent(self, model_config: core.ModelConfig) -> bcore._BackendAgent:
        return BackendAgent(model_config=model_config)


class BackendAgent(bcore._BackendAgent):

    def __init__(self, model_config: core.ModelConfig, action=None):
        """Simple constant action agent.

        Args:
            model_config: containing the gym_env_name to "train" on
            action: the action to take in all steps or None. If None no steps are taken.
        """
        super().__init__(model_config)
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
