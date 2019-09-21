"""This module contains backend core classes like Backend and BackendAgent.

    The concrete backends like tfagent or baselines are implemented in seprate modules.
"""

from abc import ABC, ABCMeta, abstractmethod
from typing import List, Optional, Tuple
import gym

from easyagents import core
from easyagents.backends import monitor
from easyagents.callbacks import plot


class _BackendEvalCallback(core.AgentCallback):
    """Evaluates an agents current policy and updates its train_context accordingly."""

    def __init__(self, train_context: core.TrainContext):
        assert train_context, "train_context not set"
        assert train_context.num_episodes_per_eval > 0, "num_episodes_per_eval is 0."

        self._train_contex = train_context

    def on_play_episode_end(self, agent_context: core.AgentContext):
        pc = agent_context.play
        tc = self._train_contex
        sum_of_r = pc.sum_of_rewards.values()
        tc.eval_rewards[tc.episodes_done_in_training] = (
            min(sum_of_r), sum(sum_of_r) / len(sum_of_r), max(sum_of_r))
        steps = [len(episode_rewards) for episode_rewards in pc.rewards.values()]
        tc.eval_steps[tc.episodes_done_in_training] = (min(steps), sum(steps) / len(steps), max(steps))


class _BackendAgent(ABC):
    """Base class for all backend agent implementations.

        Implements the train loop and calls the Callbacks.
    """

    def __init__(self, model_config: core.ModelConfig):
        assert model_config is not None, "model_config not set."

        self.model_config = model_config
        self._agent_context: core.AgentContext = core.AgentContext(self.model_config)
        self._agent_context.gym._totals = monitor._register_gym_monitor(self.model_config.original_env_name)
        self.model_config.gym_env_name = self._agent_context.gym._totals.gym_env_name

        self._preprocess_callbacks: List[core._PreProcessCallback] = [plot._PreProcess()]
        self._callbacks: List[core.AgentCallback] = []
        self._postprocess_callbacks: List[core._PostProcessCallback] = [plot._PostProcess()]

        self._train_total_episodes_on_iteration_begin: int = 0

    def _eval_current_policy(self):
        """Evaluates the current policy using play and updates the train_context

            If num_episodes_per_eval or num_iterations_per_eval is 0 no evaluation is performed.
        """
        tc = self._agent_context.train
        assert tc, "train_context not set"

        if tc.num_episodes_per_eval and tc.num_iterations_between_eval:
            callbacks = [_BackendEvalCallback(self._agent_context.train)] + self._callbacks
            self.play(play_context=core.PlayContext(self._agent_context.train), callbacks=callbacks)

    def log_api(self, api_target: str, log_msg: Optional[str] = None):
        """Logs a call to api_target with additional log_msg."""
        self._agent_context.gym._monitor_env = None
        if api_target is None:
            api_target = ''
        if log_msg is None:
            log_msg = ''
        for c in self._callbacks:
            c.on_api_log(self._agent_context, api_target, log_msg=log_msg)

    def log(self, log_msg: str):
        """Logs msg."""
        self._agent_context.gym._monitor_env = None
        if log_msg is None:
            log_msg = ''
        for c in self._callbacks:
            c.on_log(self._agent_context, log_msg=log_msg)

    def _on_gym_init_begin(self):
        """called when the monitored environment begins the instantiation of a new gym environment.

        Hint:
            the total instances count is not incremented yet."""
        self._agent_context.gym._monitor_env = None
        for c in self._callbacks:
            c.on_gym_init_begin(self._agent_context)
        self._agent_context.gym._monitor_env = None

    def _on_gym_init_end(self, env: monitor._MonitorEnv):
        """called when the monitored environment completed the instantiation of a new gym environment.

        Hint:
            o the total instances count is incremented by now
            o the new env is seeded with the api_context's seed
        """
        self._agent_context.gym._monitor_env = env
        if self._agent_context.model.seed is not None:
            self._agent_context.gym.gym_env.seed(self._agent_context.model.seed)
        for c in self._callbacks:
            c.on_gym_init_end(self._agent_context)
        self._agent_context.gym._monitor_env = None

    def _on_gym_reset_begin(self, env: monitor._MonitorEnv, **kwargs):
        """called when the monitored environment begins a reset.

        Hint:
            the total reset count is not incremented yet."""
        self._agent_context.gym._monitor_env = env
        for c in self._callbacks:
            c.on_gym_reset_begin(self._agent_context, **kwargs)
        self._agent_context.gym._monitor_env = None

    def _on_gym_reset_end(self, env: monitor._MonitorEnv, reset_result: Tuple, **kwargs):
        """called when the monitored environment completed a reset.

        Hint:
            the total episode count is incremented by now (if a step was performed before the last reset)."""
        self._agent_context.gym._monitor_env = env
        for c in self._callbacks:
            c.on_gym_reset_end(self._agent_context, reset_result, **kwargs)
        self._agent_context.gym._monitor_env = None

    def _on_gym_step_begin(self, env: monitor._MonitorEnv, action):
        """called when the monitored environment begins a step.

        Hint:
            o sets env.max_steps_per_episode if we are in train / play. Thus the episode is ended
              by the MonitorEnv if the step limit is exceeded
        """
        ac = self._agent_context
        ac.gym._monitor_env = env
        env.max_steps_per_episode = None
        if ac.is_play or ac.is_eval:
            env.max_steps_per_episode = ac.play.max_steps_per_episode
            self._on_play_step_begin(action)
        if ac.is_train:
            env.max_steps_per_episode = ac.train.max_steps_per_episode
            self._on_train_step_begin(action)
        for c in self._callbacks:
            c.on_gym_step_begin(self._agent_context, action)
        self._agent_context.gym._monitor_env = None

    def _on_gym_step_end(self, env: monitor._MonitorEnv, action, step_result: Tuple):
        """called when the monitored environment completed a step.

        Args:
            env: the gym_env the last step was done on
            step_result: the result (state, reward, done, info) of the last step call
        """
        ac = self._agent_context
        ac.gym._monitor_env = env
        if ac.is_play or ac.is_eval:
            self._on_play_step_end(action, step_result)
        if ac.is_train:
            self._on_train_step_end(action, step_result)
        for c in self._callbacks:
            c.on_gym_step_end(self._agent_context, action, step_result)
        self._agent_context.gym._monitor_env = None
        env.max_steps_per_episode = None

    def _on_play_begin(self):
        """Must NOT be called by play_implementation"""
        for c in self._callbacks:
            c.on_play_begin(self._agent_context)

    def _on_play_end(self):
        """Must NOT be called by play_implementation"""
        for c in self._callbacks:
            c.on_play_end(self._agent_context)
        self._agent_context.play.gym_env = None

    def on_play_episode_begin(self, env: gym.core.Env):
        """Must be called by play_implementation at the beginning of a new episode

        Args:
            env: the gym environment used to play the episode.
        """
        assert env, "env not set."
        assert isinstance(env, gym.core.Env), "env not an an instance of gym.Env."
        pc = self._agent_context.play
        pc.gym_env = env
        pc.steps_done_in_episode = 0
        pc.actions[pc.episodes_done + 1] = []
        pc.rewards[pc.episodes_done + 1] = []
        pc.sum_of_rewards[pc.episodes_done + 1] = 0

        for c in self._callbacks:
            c.on_play_episode_begin(self._agent_context)

    def on_play_episode_end(self):
        """Must be called by play_implementation at the end of an episode"""
        pc = self._agent_context.play
        pc.episodes_done += 1
        if pc.num_episodes and pc.episodes_done >= pc.num_episodes:
            pc.play_done = True

        for c in self._callbacks:
            c.on_play_episode_end(self._agent_context)

    def _on_play_step_begin(self, action):
        """Called before each call to gym.step on the current play env (agent_context.play.gym_env)

            Args:
                action: the action to be passed to the upcoming gym_env.step call
        """
        for c in self._callbacks:
            c.on_play_step_begin(self._agent_context, action)

    def _on_play_step_end(self, action, step_result: Tuple):
        """Called after each call to gym.step on the current play env (agent_context.play.gym_env)

        Args:
            step_result: the result (state, reward, done, info) of the last step call
        """
        (state, reward, done, info) = step_result
        pc = self._agent_context.play
        pc.steps_done_in_episode += 1
        pc.steps_done += 1
        pc.actions[pc.episodes_done + 1].append(action)
        pc.rewards[pc.episodes_done + 1].append(reward)
        pc.sum_of_rewards[pc.episodes_done + 1] += reward
        for c in self._callbacks:
            c.on_play_step_end(self._agent_context, action, step_result)

    def _on_train_begin(self):
        """Must NOT be called by train_implementation"""
        for c in self._callbacks:
            c.on_train_begin(self._agent_context)

    def _on_train_end(self):
        """Must NOT be called by train_implementation"""
        tc = self._agent_context.train
        if tc.episodes_done_in_training not in tc.eval_rewards:
            self._eval_current_policy()

        for c in self._callbacks:
            c.on_train_end(self._agent_context)

    def on_train_iteration_begin(self):
        """Must be called by train_implementation at the begining of a new iteration"""
        tc = self._agent_context.train
        tc.episodes_done_in_iteration = 0
        tc.steps_done_in_iteration = 0
        if tc.iterations_done_in_training == 0:
            self._eval_current_policy()
        self._train_total_episodes_on_iteration_begin = self._agent_context.gym._totals.episodes_done

        for c in self._callbacks:
            c.on_train_iteration_begin(self._agent_context)

    def on_train_iteration_end(self, loss: float, **kwargs):
        """Must be called by train_implementation at the end of an iteration

        Evaluates the current policy. Use kwargs to set additional dict values in train context.
        Eg for an ActorCriticTrainContext the losses may be set like this:
            on_train_iteration(loss=123,actor_loss=456,critic_loss=789)

        Args:
            loss: loss after the training of the model in this iteration
            **kwargs: if a keyword matches a dict property of the TrainContext instance, then
                        the dict[episodes_done_in_training] is set to the arg.
        """
        tc = self._agent_context.train
        totals = self._agent_context.gym._totals
        tc.episodes_done_in_iteration = (totals.episodes_done - self._train_total_episodes_on_iteration_begin)
        tc.episodes_done_in_training += tc.episodes_done_in_iteration
        tc.loss[tc.episodes_done_in_training] = loss

        # set traincontext dict from kwargs:
        for prop_name in kwargs:
            prop_instance = getattr(tc, prop_name, None)
            prop_value = kwargs[prop_name]
            if prop_instance is not None and isinstance(prop_instance, dict):
                prop_instance[tc.episodes_done_in_training] = prop_value

        tc.iterations_done_in_training += 1
        if tc.num_iterations is not None:
            tc.training_done = tc.iterations_done_in_training >= tc.num_iterations
        self._train_total_episodes_on_iteration_begin = 0
        if tc.num_iterations_between_eval and (tc.iterations_done_in_training % tc.num_iterations_between_eval == 0):
            self._eval_current_policy()

        for c in self._callbacks:
            c.on_train_iteration_end(self._agent_context)

    def _on_train_step_begin(self, action):
        """Called before each call to gym.step on the current train env (agent_context.train.gym_env)

            Args:
                action: the action to be passed to the upcoming gym_env.step call
        """
        pass

    # noinspection PyUnusedLocal
    def _on_train_step_end(self, action: object, step_result: Tuple):
        """Called after each call to gym.step on the current train env (agent_context.train.gym_env)

        Args:
            step_result: the result (state, reward, done, info) of the last step call
        """
        tc = self._agent_context.train
        tc.steps_done_in_iteration += 1
        tc.steps_done_in_training += 1

    def play(self, play_context: core.PlayContext, callbacks: List[core.AgentCallback]):
        """Forwarding to play_implementation overriden by the subclass.

            Args:
                play_context: play configuration to be used
                callbacks: list of callbacks called during play.
        """
        assert callbacks is not None, "callbacks not set"
        assert play_context, "play_context not set"
        assert self._agent_context.play is None, "play_context already set in agent_context"

        play_context._reset()
        play_context._validate()
        self._agent_context.play = play_context
        old_callbacks = self._callbacks
        self._callbacks = callbacks
        try:
            monitor._MonitorEnv._register_backend_agent(self)
            self._on_play_begin()
            self.play_implementation(self._agent_context.play)
            self._on_play_end()
        finally:
            monitor._MonitorEnv._register_backend_agent(None)
            self._callbacks = old_callbacks
            self._agent_context.play = None

    @abstractmethod
    def play_implementation(self, play_context: core.PlayContext):
        """Agent specific implementation of playing a single episode with the current policy.

            For implementation details see BackendBaseAgent.
        """

    def train(self, train_context: core.TrainContext, callbacks: List[core.AgentCallback]):
        """Forwarding to train_implementation overriden by the subclass

            Args:
                train_context: training configuration to be used
                callbacks: list of callbacks called during the training and evaluation.
        """
        assert callbacks is not None, "callbacks not set"
        assert train_context, "train_context not set"

        train_context._reset()
        train_context._validate()
        self._agent_context.train = train_context
        self._agent_context.play = None
        self._callbacks = callbacks

        try:
            monitor._MonitorEnv._register_backend_agent(self)
            self._on_train_begin()
            self.train_implementation(self._agent_context.train)
            self._on_train_end()
        finally:
            monitor._MonitorEnv._register_backend_agent(None)
            self._callbacks = None
            self._agent_context.play = None
            self._agent_context.train = None

    @abstractmethod
    def train_implementation(self, train_context: core.TrainContext):
        """Agent specific implementation of the train loop.

            For implementational details see BackendBaseAgent.
        """


class BackendAgent(_BackendAgent, metaclass=ABCMeta):
    """Base class for all BackendAgent implementation.

        Explicitely exhibits all methods that should be overriden by an implementing agent.
    """

    @abstractmethod
    def play_implementation(self, play_context: core.PlayContext):
        """Agent specific implementation of playing a number of episodes with the current policy.

            The implementation should have the form:

            while True:
                on_play_episode_begin(env)
                state = env.reset()
                while True:
                    action = _trained_policy.action(state)
                    (state, reward, done, info) = env.step(action)
                    if done:
                        break
                on_play_episode_end()
                if play_context.play_done:
                    break

            Args:
                play_context: play configuration to be used
        """

    @abstractmethod
    def train_implementation(self, train_context: core.TrainContext):
        """Agent specific implementation of the train loop.

            The implementation should have the form:

            while True:
                on_iteration_begin
                for e in num_episodes_per_iterations
                    play episode and record steps (while steps_in_episode < max_steps_per_episode and)
                train policy for num_epochs_per_iteration epochs
                on_iteration_end( loss )
                if training_done
                    break

            Args:
                train_context: context configuring the train loop

            Hints:
            o the subclasses training loss is passed through to BackendAgent by on_iteration_end.
              Thus the subclass must not add the experienced loss to the TrainContext.
        """


class BackendAgentFactory(ABC):
    """Backend agent factory defining the currently available agents (algorithms).
    """

    name: str = 'abstract_BackendAgentFactory'

    def create_dqn_agent(self, model_config: core.ModelConfig) -> _BackendAgent:
        """Create an instance of DqnAgent wrapping this backends implementation.

            If this backend does not implement DqbAgent then throw a NotImplementedError exception.

        Args:
            model_config: the agents configuration containing in patricular the name of the gym environment
                to be used and the nn architecture.
        """
        raise NotImplementedError(f'DqnAgent not implemented by backend "{self.name}"')

    def create_ppo_agent(self, model_config: core.ModelConfig) -> _BackendAgent:
        """Create an instance of PpoAgent wrapping this backends implementation.

            If this backend does not implement PpoAgent then throw a NotImplementedError exception.

        Args:
            model_config: the agents configuration containing in patricular the name of the gym environment
                to be used and the nn architecture.
        """
        raise NotImplementedError(f'PpoAgent not implemented by backend "{self.name}"')

    def create_random_agent(self, model_config: core.ModelConfig) -> _BackendAgent:
        """Create an instance of RandomAgent wrapping this backends implementation.

            If this backend does not implement RandomAgent then throw a NotImplementedError exception.

        Args:
            model_config: the agents configuration containing in patricular the name of the gym environment
                to be used and the nn architecture.

        Hints:
            o play() should be callable without a prior call to train()
            o train() should evaluate in each iteration 1 episode, guaranteeing that the results of each
              iteration evaluation (train_context.eval_rewards, train_context.eval_steps) is stored in a
              seperate episode dictionary entry
        """
        raise NotImplementedError(f'RandomAgent not implemented by backend "{self.name}"')

    def create_reinforce_agent(self, model_config: core.ModelConfig) -> _BackendAgent:
        """Create an instance of ReinforceAgent wrapping this backends implementation.

            If this backend does not implement PpoAgent then throw a NotImplementedError exception.

        Args:
            model_config: the agents configuration containing in patricular the name of the gym environment
                to be used and the nn architecture.
        """
        raise NotImplementedError(f'ReinforceAgent not implemented by backend "{self.name}"')
