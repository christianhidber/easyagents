"""This module contains backend core classes like Backend and BackendAgent.

    The concrete backends like tfagent or baselines are implemented in seprate modules.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Callable
import logging

import gym.core
from easyagents import core
from easyagents.backends import monitor


class BackendAgent(ABC):
    """Base class for all backend agent implementations.

        Implements the train loop and calls the TrainCallbacks.
    """

    def __init__(self, model_config: core.ModelConfig):
        assert model_config is not None, "model_config not set."
        self.model_config = model_config
        self._train_callbacks: Optional[List[core.TrainCallback]] = None
        self.train_context: Optional[core.TrainContext] = None
        self._play_callbacks: Optional[List[core.PlayCallback]] = None
        self._play_context: Optional[core.PlayContext] = None
        self._api_callbacks: Optional[List[core.ApiCallback]] = None
        self._api_context: Optional[core.ApiContext] = None
        self._total = monitor._register_gym_monitor(model_config.original_env_name)
        self.gym_env_name = self._total.gym_env_name

        self._total_episodes_on_iteration_begin = 0
        self._total_steps_on_iteration_begin = 0

    def api_call(self, api_id: str, f: Callable):
        """Called by the implementing agent to execute a call into the backend library.

            This enables detailed inspection of the agents interactions with its underlying implementation.
        """
        self._api_context.api_id = api_id
        for c in self._api_callbacks:
            c.on_backend_call_begin(self._api_context)
        result = f()
        for c in self._api_callbacks:
            c.on_backend_call_end(self._api_context)
        self._api_context.api_id = None
        return result

    def api_log(self, msg: str):
        """Logs msg.

            Hint:
            o may be propgated to the api_callbacks in a later version (19Q3)
        """
        logging.debug(msg)

    def _on_gym_init_begin(self):
        """called when the monitored environment begins the instantiation of a new gym environment.

        Hint:
            the total instances count is not incremented yet."""
        self._api_context.gym_env = None
        for c in self._api_callbacks:
            c.on_gym_init_begin(self._api_context)
        self._api_context.gym_env = None

    def _on_gym_init_end(self, gym_env: gym.core.Env):
        """called when the monitored environment completed the instantiation of a new gym environment.

        Hint:
            o the total instances count is incremented by now
            o the new env is seeded with the api_context's seed
        """
        self._api_context.gym_env = gym_env
        if self.model_config.seed is not None:
            self._api_context.gym_env.seed(self.model_config.seed)
        for c in self._api_callbacks:
            c.on_gym_init_end(self._api_context)
        self._api_context.gym_env = None

    def _on_gym_reset_begin(self, gym_env: gym.core.Env, **kwargs):
        """called when the monitored environment begins a reset.

        Hint:
            the total reset count is not incremented yet."""
        self._api_context.gym_env = gym_env
        for c in self._api_callbacks:
            c.on_gym_reset_begin(self._api_context)
        self._api_context.gym_env = None

    def _on_gym_reset_end(self, gym_env: gym.core.Env, reset_result: Tuple, **kwargs):
        """called when the monitored environment completed a reset.

        Hint:
            the total episode count is incremented by now (if a step was performed before the last reset)."""
        self._api_context.gym_env = gym_env
        for c in self._api_callbacks:
            c.on_gym_reset_end(self._api_context, reset_result, **kwargs)
        self._api_context.gym_env = None

    def _on_gym_step_begin(self, gym_env: gym.core.Env, action):
        """called when the monitored environment begins a step.

        Hint:
            the total step count is not incremented yet."""
        self._api_context.gym_env = gym_env
        for c in self._api_callbacks:
            c.on_gym_step_begin(self._api_context, action)
        self._api_context.gym_env = None

    def _on_gym_step_end(self, gym_env: gym.core.Env, action, step_result: Tuple):
        """called when the monitored environment completed a step.

        Hint:
            the step count is incremented by now."""
        self._api_context.gym_env = gym_env
        for c in self._api_callbacks:
            c.on_gym_step_end(self._api_context, action, step_result)
        self._api_context.gym_env = None

    def on_iteration_begin(self):
        """Must be called by train_implementation at the begining of a new iteration"""
        self._total_episodes_on_iteration_begin = self._total.episodes_done
        self._total_steps_on_iteration_begin = self._total.steps_done
        self.train_context.episodes_done_in_iteration = 0
        self.train_context.steps_done_in_iteration = 0

        for c in self._train_callbacks:
            c.on_iteration_begin(self.train_context)

    def on_iteration_end(self, iteration_loss: float):
        """Must be called by train_implementation at the end of an iteration

        Args:
            iteration_loss: loss after the training of the model in this iteration
        """
        self.train_context.episodes_done_in_iteration = (self._total.episodes_done -
                                                         self._total_episodes_on_iteration_begin)
        self.train_context.episodes_done_in_training += self.train_context.episodes_done_in_iteration
        self.train_context.steps_done_in_iteration = (self._total.steps_done - self._total_steps_on_iteration_begin)
        self.train_context.steps_done_in_training += self.train_context.steps_done_in_iteration
        self.train_context.loss[self.train_context.episodes_done_in_training] = iteration_loss

        for c in self._train_callbacks:
            c.on_iteration_end(self.train_context)

        self.train_context.iterations_done_in_training += 1

    def _on_train_begin(self):
        """Must NOT be called by train_implementation"""

        self._total_episodes_on_iteration_begin = 0
        self._total_steps_on_iteration_begin = 0

        for c in self._train_callbacks:
            c.on_train_begin(self.train_context)

    def _on_train_end(self):
        """Must NOT be called by train_implementation"""
        for c in self._train_callbacks:
            c.on_train_end(self.train_context)

    def train(self,
              train_callbacks: List[core.TrainCallback],
              train_context: core.TrainContext,
              play_callbacks: List[core.PlayCallback],
              api_callbacks: List[core.ApiCallback]
              ):
        """
            Minimal implementation forwarding to train_implementation overriden by the subclass

            Args:
                train_context: training configuration to be used
                train_callbacks: list of callbacks called during the train loop (but not during the intermittent evals)
                play_callbacks: list of callbacks called during the eval (but not during data collection for training)
                api_callbacks: list of callbacks called during backend and gym api calls
        """
        assert train_callbacks is not None, "train_callbacks not set"
        assert train_context, "train_context not set"
        assert play_callbacks is not None, "play_callbacks not set"
        assert api_callbacks is not None, "api_callbacks not set"

        self._train_callbacks = train_callbacks
        self.train_context = train_context
        self.train_context._reset()
        self.train_context._validate()
        self._play_callbacks = play_callbacks
        self._play_context = core.PlayContext(train_context)
        self._api_callbacks = api_callbacks
        self._api_context = core.ApiContext()

        try:
            monitor._MonitorEnv._register_backend_agent(self)
            self._on_train_begin()
            self.train_implementation(self.train_context)
            self._on_train_end()
        finally:
            monitor._MonitorEnv._register_backend_agent(None)

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
