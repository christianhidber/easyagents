"""This module contains the backend implementation for tensorforce

    see https://github.com/tensorforce/tensorforce
"""
import datetime
import math
import os
import tempfile
from abc import ABCMeta
from typing import List, Dict, Optional, Type

import gym
from tensorforce.agents import Agent
from tensorforce.environments import Environment
from tensorforce.execution import Runner

import easyagents.backends.core


class TforceAgent(easyagents.backends.core.BackendAgent, metaclass=ABCMeta):
    """ Base class for agents based on the tensorforce implementation."""

    def __init__(self, model_config: easyagents.core.ModelConfig):
        """
        Args:
            model_config: the model configuration including the name of the target gym environment
                as well as the neural network architecture.
        """
        super().__init__(model_config=model_config, backend_name=TensorforceAgentFactory.backend_name)
        self._agent: Optional[Agent] = None
        self._play_env: Optional[Environment] = None

    def _create_env(self) -> Environment:
        """Creates a tensorforce Environment encapsulating the underlying gym environment given in self.model_config"""
        self.log_api('Environment.create', f'(environment="gym", level={self.model_config.original_env_name})')
        result = Environment.create(environment='gym', level=self.model_config.gym_env_name)
        return result

    def _create_network_specification(self):
        """Creates a tensorforce network specification based on the layer specification given in self.model_config"""
        result: List[Dict] = []
        layer_sizes = self.model_config.fc_layers
        for layer_size in layer_sizes:
            result.append(dict(type='dense', size=layer_size, activation='relu'))
        return result

    def _get_temp_path(self):
        result = os.path.join(tempfile.gettempdir(), tempfile.gettempprefix())
        n = datetime.datetime.now()
        result = result + f'-{n.year % 100:2}{n.month:02}{n.day:02}-{n.hour:02}{n.minute:02}{n.second:02}-' + \
                 f'{n.microsecond:06}'
        return result

    def _train_with_runner(self, train_env: Environment, train_context: easyagents.core.TrainContext):
        """Trains the self._agent using a tensorforce runner.

        Args:
            train_env: the tensorforce environment to use for the training
            train_context: context containing the training parameters
        """
        assert train_context
        assert train_env
        assert self._agent

        def step_callback(tforce_runner: Runner) -> bool:
            """
                Returns:
                    true if the training should continue, false to end the training
            """
            result = not train_context.training_done
            if result:
                is_iteration_end = False
                if isinstance(train_context, easyagents.core.EpisodesTrainContext):
                    ec: easyagents.core.EpisodesTrainContext = train_context
                    current_episode = tforce_runner.episodes - 1
                    current_step = tforce_runner.episode_timestep
                    is_iteration_end = current_episode > 0 and \
                                       current_step == 1 and \
                                       current_episode % ec.num_episodes_per_iteration == 0
                elif isinstance(train_context, easyagents.core.StepsTrainContext):
                    sc: easyagents.core.StepsTrainContext = train_context
                    current_step = tforce_runner.timesteps - 1
                    is_iteration_end = current_step > 0 and \
                                       current_step % sc.num_steps_per_iteration == 0
                else:
                    raise AssertionError("Unexpected TrainContext detected.")
                if is_iteration_end:
                    self.on_train_iteration_end(loss=math.nan, actor_loss=math.nan, critic_loss=math.nan)
                    result = not train_context.training_done
                    if result:
                        self.on_train_iteration_begin()
            return result

        def eval_callback(tforce_runner: Runner) -> bool:
            """
                This call back should never be called. Raises an exception.

                Policy evaluation is performed by easyagents directly.
            """
            raise AssertionError("unexpected eval_callback call.")

        # Initialize the runner
        self.log_api('Runner.create', "(agent=..., environment=...)")
        runner = Runner(agent=self._agent, environment=train_env)

        # Start the runner
        self.on_train_iteration_begin()
        self.log_api('runner.run',
                     f'(num_episodes=None, use_tqdm=False, callback=..., callback_timestep_frequency=1, ' +
                     'evaluation_callback=..., evaluation_frequency=None, evaluation=False, ' +
                     'num_evaluation_iterations=0)')
        runner.run(num_episodes=None,
                   use_tqdm=False,
                   callback=step_callback,
                   callback_timestep_frequency=1,
                   evaluation_callback=eval_callback,
                   evaluation_frequency=None,
                   evaluation=False,
                   num_evaluation_iterations=0
                   )
        assert train_context.training_done
        "Unexpected runner termination."
        runner.close()

    def load_implementation(self, directory: str):
        """Loads a previously trained and saved actor policy from directory.

        The loaded policy may afterwards be used by calling play().

        Args:
            directory: the directory containing the trained policy.
        """
        pass

    def play_implementation(self, play_context: easyagents.core.PlayContext):
        """Agent specific implementation of playing a single episode with the current policy.

            Args:
                play_context: play configuration to be used
        """
        assert play_context, "play_context not set."
        assert self._agent, "agent not set. call train first."

        if self._play_env is None:
            self._play_env = self._create_env()
        while True:
            # noinspection PyUnresolvedReferences
            gym_env: gym.Env = self._play_env.environment
            self.on_play_episode_begin(env=gym_env)
            state = self._play_env.reset()
            done = False
            while not done:
                action = self._agent.act(state, evaluation=True)
                state, terminal, reward = self._play_env.execute(actions=action)
                if isinstance(terminal, bool):
                    done = terminal
                else:
                    done = terminal > 0
            self.on_play_episode_end()
            if play_context.play_done:
                break
        return

    def save_implementation(self, directory: str):
        """Agent speecific implementation of saving the weights for the actor policy.

        Save must only guarantee to persist the weights of the actor policy.
        The implementation may write multiple files with fixed filenames.

        Args:
             directory: the directory to save the policy weights to.
        """


class TforceDqnAgent(TforceAgent):
    """ Agent based on the DQN algorithm using the tensorforce implementation."""

    def __init__(self, model_config: easyagents.core.ModelConfig,
                 enable_dueling_dqn: bool = False):
        """
        Args:
            model_config: the model configuration including the name of the target gym environment
                as well as the neural network architecture.
        """
        super().__init__(model_config=model_config)

    def train_implementation(self, train_context: easyagents.core.StepsTrainContext):
        """Tensorforce Dqn Implementation of the train loop.

            The implementation follows https://github.com/tensorforce/tensorforce/blob/master/examples/quickstart.py
        """
        tc = train_context
        train_env = self._create_env()
        network = self._create_network_specification()

        agent_type = 'dqn'
        self.log_api('Agent.create',
                     f'(agent="{agent_type}", ' +
                     f'network={network}, ' +
                     f'memory={tc.max_steps_in_buffer}, ' +
                     f'start_updating={tc.num_steps_buffer_preload},'
                     f'learning_rate={tc.learning_rate}, ' +
                     f'batch_size={tc.num_steps_sampled_from_buffer}, ' +
                     f'update_frequeny={tc.num_steps_per_iteration}, ' +
                     f'discount={tc.reward_discount_gamma})')
        self._agent = Agent.create(
            agent=agent_type,
            environment=train_env,
            network=network,
            memory=tc.max_steps_in_buffer,
            start_updating=tc.num_steps_buffer_preload,
            learning_rate=tc.learning_rate,
            batch_size=tc.num_steps_sampled_from_buffer,
            update_frequency=tc.num_steps_per_iteration,
            discount=tc.reward_discount_gamma,
        )
        self._train_with_runner(train_env, tc)


class TforceDuelingDqnAgent(TforceDqnAgent):
    """ Agent based on the DQN algorithm using the tensorforce implementation."""

    def __init__(self, model_config: easyagents.core.ModelConfig):
        """
        Args:
            model_config: the model configuration including the name of the target gym environment
                as well as the neural network architecture.
        """
        super().__init__(model_config=model_config, enable_dueling_dqn=True)


class TforcePpoAgent(TforceAgent):
    """ Agent based on the PPO algorithm using the tensorforce implementation."""

    def train_implementation(self, train_context: easyagents.core.PpoTrainContext):
        """Tensorforce Ppo Implementation of the train loop.

            The implementation follows https://github.com/tensorforce/tensorforce/blob/master/examples/quickstart.py
        """
        tc = train_context
        train_env = self._create_env()
        network = self._create_network_specification()

        self.log_api('Agent.create', f'(agent="ppo", environment=..., ' +
                     f'network={network}' +
                     f'learning_rate={tc.learning_rate}, ' +
                     f'batch_size={tc.num_episodes_per_iteration}, ' +
                     f'optimization_steps={tc.num_epochs_per_iteration}, ' +
                     f'discount={tc.reward_discount_gamma})')
        self._agent = Agent.create(
            agent='ppo',
            environment=train_env,
            network=network,
            learning_rate=tc.learning_rate,
            batch_size=tc.num_episodes_per_iteration,
            optimization_steps=tc.num_epochs_per_iteration,
            discount=tc.reward_discount_gamma,
        )
        self._train_with_runner(train_env, tc)


class TforceRandomAgent(TforceAgent):
    """ Random agent using the tensorforce implementation."""

    def train_implementation(self, train_context: easyagents.core.TrainContext):
        assert isinstance(train_context, easyagents.core.TrainContext)
        train_env = self._create_env()
        self.log_api('Agent.create', f'(agent="random", environment=...)')
        self._agent = Agent.create(agent='random', environment=train_env)
        if not self._agent.model.is_initialized:
            self._agent.initialize()

        while not train_context.training_done:
            self.on_train_iteration_begin()
            state = train_env.reset()
            done = False
            while not done:
                action = self._agent.act(state, evaluation=True)
                state, terminal, reward = train_env.execute(actions=action)
                if isinstance(terminal, bool):
                    done = terminal
                else:
                    done = terminal > 0
            self.on_train_iteration_end(math.nan)


class TforceReinforceAgent(TforceAgent):
    """ Agent based on the REINFORCE algorithm using the tensorforce implementation."""

    def train_implementation(self, train_context: easyagents.core.EpisodesTrainContext):
        """Tensorforce REINFORCE Implementation of the train loop.

            The implementation follows https://github.com/tensorforce/tensorforce/blob/master/examples/quickstart.py
        """
        assert isinstance(train_context, easyagents.core.EpisodesTrainContext)
        tc = train_context
        train_env = self._create_env()
        network = self._create_network_specification()

        self.log_api('Agent.create', f'(agent="vpg", environment=..., ' +
                     f'network={network}' +
                     f'learning_rate={tc.learning_rate}, ' +
                     f'batch_size={tc.num_episodes_per_iteration}, ' +
                     f'discount={tc.reward_discount_gamma})')
        self._agent = Agent.create(
            agent='vpg',
            environment=train_env,
            network=network,
            learning_rate=tc.learning_rate,
            batch_size=tc.num_episodes_per_iteration,
            discount=tc.reward_discount_gamma,
        )
        self._train_with_runner(train_env, tc)


class TensorforceAgentFactory(easyagents.backends.core.BackendAgentFactory):
    """Backend for Tensorforce.

        Serves as a factory to create algorithm specific wrappers for the tensorforce implementations.
    """

    backend_name: str = 'tensorforce'

    def get_algorithms(self) -> Dict[Type, Type[easyagents.backends.core.BackendAgent]]:
        """Yields a mapping of EasyAgent types to the implementations provided by this backend."""
        return {
            easyagents.agents.DqnAgent: TforceDqnAgent,
            easyagents.agents.PpoAgent: TforcePpoAgent,
            easyagents.agents.RandomAgent: TforceRandomAgent,
            easyagents.agents.ReinforceAgent: TforceReinforceAgent}
