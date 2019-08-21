"""This module contains backend core classes like Backend and BackendAgent.

    The concrete backends like tfagent or baselines are implemented in seprate modules.
"""

from abc import ABC, abstractmethod
from typing import List, Callable, Optional

from easyagents import core


class BackendAgent(ABC):
    """Base class for all backend agent implementations.

        Implements the train loop and calls the TrainCallbacks.
    """

    def __init__(self, model_config: core.ModelConfig):
        assert model_config is not None, "model_config not set."
        self._model_config = model_config
        self._train_context: Optional[core.TrainContext] = None
        self._train_callbacks: Optional[List[core.TrainCallback]] = None
        self._play_callbacks: Optional[List[core.PlayCallback]] = None
        self._api_callbacks: Optional[List[core.ApiCallback]] = None

    def on_iteration_begin(self):
        """Must be called by train_implementation at the begining of a new iteration"""
        for c in self._train_callbacks:
            c.on_iteration_begin(self._train_context)

    def on_iteration_end(self, iteration_loss: float):
        """Must be called by train_implementation at the end of an iteration

        Args:
            iteration_loss: loss after the training of the model in this iteration
        """
        self._train_context.loss[self._train_context.current_episode] = iteration_loss
        for c in self._train_callbacks:
            c.on_iteration_end(self._train_context)

    def _on_train_begin(self):
        """Must NOT be called by train_implementation"""
        for c in self._train_callbacks:
            c.on_train_begin(self._train_context)

    def _on_train_end(self):
        """Must NOT be called by train_implementation"""
        for c in self._train_callbacks:
            c.on_train_end(self._train_context)

    def train(self,
              train_context: core.TrainContext,
              train_callbacks: List[core.TrainCallback],
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
        assert train_context, "train_context not set"
        assert train_callbacks is not None, "train_callbacks not set"
        assert play_callbacks is not None, "play_callbacks not set"
        assert api_callbacks is not None, "api_callbacks not set"

        self._train_context = train_context
        self._train_context._reset()
        self._train_context._validate()
        self._train_callbacks = train_callbacks
        self._play_callbacks = play_callbacks
        self._api_callbacks = api_callbacks

        self._on_train_begin()
        self.train_implementation(self._train_context)
        self._on_train_end()

    @abstractmethod
    def train_implementation(self, train_context: core.TrainContext):
        """Agent specific implementation of the train loop inside the subclassing agent.

            The implementation should have the form:

            on_train_begin
            for i in num_iterations
                on_iteration_begin
                for e in num_episodes_per_iterations
                    play episode and record steps (while steps_in_episode < max_steps_per_episode and)
                train policy for num_epochs_per_iteration epochs
                on_iteration_end( loss )
                if training_done
                    break
            on_train_end

            Hints:
            o the subclasses training loss is passed through to BackendAgent by on_iteration_end.
              Thus the subclass must not add the experienced loss to the TrainContext.
        """


class Backend(ABC):
    """Abstract base class for all backends.

    It contains in particular the code shared between all backends.
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
