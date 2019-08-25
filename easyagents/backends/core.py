"""This module contains backend core classes like Backend and BackendAgent.

    The concrete backends like tfagent or baselines are implemented in seprate modules.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import logging

import gym.core
from easyagents import core
from easyagents.backends import monitor


class BackendAgent(ABC):
    """Base class for all backend agent implementations.

        Implements the train loop and calls the Callbacks.
    """

    def __init__(self, model_config: core.ModelConfig):
        assert model_config is not None, "model_config not set."

        self.model_config = model_config
        self._callbacks: Optional[List[core.AgentCallback]] = []
        self._agent_context: core.AgentContext = core.AgentContext(self.model_config)
        self._agent_context.api._totals = monitor._register_gym_monitor(self.model_config.original_env_name)
        self.model_config.gym_env_name = self._agent_context.api._totals.gym_env_name

        self._total_episodes_on_iteration_begin = 0
        self._total_steps_on_iteration_begin = 0

    def api_log(self, api_target: str, log_msg: Optional[str] = None):
        """Logs a call to api_target with additional log_msg."""
        self._agent_context.api.gym_env = None
        if api_target is None:
            api_target=''
        if log_msg is None:
            log_msg=''
        for c in self._callbacks:
            c.on_api_log(self._agent_context, api_target, log_msg=log_msg)

    def log(self, log_msg: str):
        """Logs msg."""
        self._agent_context.api.gym_env = None
        if log_msg is None:
            log_msg=''
        for c in self._callbacks:
            c.on_log(self._agent_context, log_msg=log_msg)

    def _on_gym_init_begin(self):
        """called when the monitored environment begins the instantiation of a new gym environment.

        Hint:
            the total instances count is not incremented yet."""
        self._agent_context.api.gym_env = None
        for c in self._callbacks:
            c.on_gym_init_begin(self._agent_context)
        self._agent_context.api.gym_env = None

    def _on_gym_init_end(self, gym_env: gym.core.Env):
        """called when the monitored environment completed the instantiation of a new gym environment.

        Hint:
            o the total instances count is incremented by now
            o the new env is seeded with the api_context's seed
        """
        self._agent_context.api.gym_env = gym_env
        if self._agent_context.model.seed is not None:
            self._agent_context.api.gym_env.seed(self._agent_context.model.seed)
        for c in self._callbacks:
            c.on_gym_init_end(self._agent_context)
        self._agent_context.api.gym_env = None

    def _on_gym_reset_begin(self, gym_env: gym.core.Env, **kwargs):
        """called when the monitored environment begins a reset.

        Hint:
            the total reset count is not incremented yet."""
        self._agent_context.api.gym_env = gym_env
        for c in self._callbacks:
            c.on_gym_reset_begin(self._agent_context, **kwargs)
        self._agent_context.api.gym_env = None

    def _on_gym_reset_end(self, gym_env: gym.core.Env, reset_result: Tuple, **kwargs):
        """called when the monitored environment completed a reset.

        Hint:
            the total episode count is incremented by now (if a step was performed before the last reset)."""
        self._agent_context.api.gym_env = gym_env
        for c in self._callbacks:
            c.on_gym_reset_end(self._agent_context, reset_result, **kwargs)
        self._agent_context.api.gym_env = None

    def _on_gym_step_begin(self, gym_env: gym.core.Env, action):
        """called when the monitored environment begins a step.

        Hint:
            the total step count is not incremented yet."""
        self._agent_context.api.gym_env = gym_env
        for c in self._callbacks:
            c.on_gym_step_begin(self._agent_context, action)
        self._agent_context.api.gym_env = None

    def _on_gym_step_end(self, gym_env: gym.core.Env, action, step_result: Tuple):
        """called when the monitored environment completed a step.

        Hint:
            the step count is incremented by now."""
        self._agent_context.api.gym_env = gym_env
        for c in self._callbacks:
            c.on_gym_step_end(self._agent_context, action, step_result)
        self._agent_context.api.gym_env = None

    def _on_train_begin(self):
        """Must NOT be called by train_implementation"""

        self._total_episodes_on_iteration_begin = 0
        self._total_steps_on_iteration_begin = 0

        for c in self._callbacks:
            c.on_train_begin(self._agent_context)

    def _on_train_end(self):
        """Must NOT be called by train_implementation"""
        for c in self._callbacks:
            c.on_train_end(self._agent_context)

    def on_train_iteration_begin(self):
        """Must be called by train_implementation at the begining of a new iteration"""
        self._total_episodes_on_iteration_begin = self._agent_context.api._totals.episodes_done
        self._total_steps_on_iteration_begin = self._agent_context.api._totals.steps_done
        self._agent_context.train.episodes_done_in_iteration = 0
        self._agent_context.train.steps_done_in_iteration = 0

        for c in self._callbacks:
            c.on_train_iteration_begin(self._agent_context)

    def on_train_iteration_end(self, iteration_loss: float):
        """Must be called by train_implementation at the end of an iteration

        Args:
            iteration_loss: loss after the training of the model in this iteration
        """
        tc = self._agent_context.train
        totals=self._agent_context.api._totals
        tc.episodes_done_in_iteration = (totals.episodes_done - self._total_episodes_on_iteration_begin)
        tc.episodes_done_in_training += tc.episodes_done_in_iteration
        tc.steps_done_in_iteration = (totals.steps_done - self._total_steps_on_iteration_begin)
        tc.steps_done_in_training += tc.steps_done_in_iteration
        tc.loss[tc.episodes_done_in_training] = iteration_loss
        tc.iterations_done_in_training += 1
        if tc.num_iterations is not None:
            tc.training_done = tc.iterations_done_in_training >= tc.num_iterations

        for c in self._callbacks:
            c.on_train_iteration_end(self._agent_context)


    def train(self, train_context: core.TrainContext, callbacks: List[core.AgentCallback]):
        """
            Minimal implementation forwarding to train_implementation overriden by the subclass

            Args:
                train_context: training configuration to be used
                callbacks: list of callbacks called during the training and evaluation.
        """
        assert callbacks is not None, "callbacks not set"
        assert train_context, "train_context not set"

        train_context._reset()
        train_context._validate()
        self._agent_context.train = train_context
        self._agent_context.play = core.PlayContext(train_context=train_context)
        self._callbacks = callbacks

        try:
            monitor._MonitorEnv._register_backend_agent(self)
            self._on_train_begin()
            self.train_implementation(self._agent_context.train)
            self._on_train_end()
        finally:
            monitor._MonitorEnv._register_backend_agent(None)
            self._callbacks = None

    @abstractmethod
    def train_implementation(self, train_context: core.TrainContext):
        """Agent specific implementation of the train loop inside the subclassing agent.

            The implementation should have the form:

            for i in num_iterations
                on_iteration_begin
                for e in num_episodes_per_iterations
                    play episode and record steps (while steps_in_episode < max_steps_per_episode and)
                train policy for num_epochs_per_iteration epochs
                on_iteration_end( loss )
                if training_done
                    break

            Hints:
            o the subclasses training loss is passed through to BackendAgent by on_iteration_end.
              Thus the subclass must not add the experienced loss to the TrainContext.
        """


class BackendAgentFactory(ABC):
    """Backend agent factory defining the currently available agents (algorithms).
    """

    @abstractmethod
    def create_ppo_agent(self, model_config: core.ModelConfig) -> BackendAgent:
        """Create an instance of PpoAgent wrapping this backends implementation.

            If this backend does not implement PpoAgent then throw a NotImplementedError exception.

        Args:
            model_config: the agents configuration containing in patricular the name of the gym environment
                to be used and the nn architecture.
        """
        pass
