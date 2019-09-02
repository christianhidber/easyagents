"""This module contains the core datastructures shared between fronten and backend like the definition
    of all callbacks and agent configurations.
"""

from abc import ABC
from typing import Optional, Dict, Tuple, List

import easyagents.env
import gym.core
import matplotlib.pyplot as plt

"""The seed used for all agents and gym environments. If None no seed is set (default)."""
seed: Optional[int] = None


class GymContext(object):
    """Contains the context for gym api call

    Attributes:
        gym_env: the target gym instance of a pending gym api call
    """

    def __init__(self):
        self.gym_env: Optional[gym.core.Env] = None
        self._totals = None

    def __str__(self):
        return f'{self._totals}'


class PyPlotContext(object):
    """Contain the context for the maplotlib.pyplot figure plotting.

    Attributes
        figure: the figure to plot to
        is_jupyter_active: True if we plot to jupyter notebook cell, False otherwise.
        max_columns: the max number of subplot columns in the pyplot figure
    """

    def __init__(self):
        self.figure: Optional[plt.Figure] = None
        self._call_jupyter_display = False
        self.is_jupyter_active = False
        self.max_columns = 3

    def __str__(self):
        return f'is_jupyter_active={self.is_jupyter_active} ' +\
                f'max_columns={self.max_columns}'

    @property
    def is_plot_active(self):
        """Yields true if the figure contains at least 1 subplot, false otherweise."""
        return len(self.figure.axes) > 0

class ModelConfig(object):
    """The model configurations, containing the name of the gym environment and the neural network architecture.

        Attributes:
            original_env_name: the name of the underlying gym environment, eg 'CartPole-v0'
            gym_env_name: the name of the actual gym environment used (a wrapper around the environment given
                by original_env_name)
            fc_layers: int tuple defining the number and size of each fully connected layer.
            seed: the seed to be used for example for the gym_env or None for no seed
    """

    def __init__(self, gym_env_name: str, fc_layers: Optional[Tuple[int, ...]] = None):
        """
            Args:
                gym_env_name: the name of the registered gym environment to use, eg 'CartPole-v0'
                fc_layers: int tuple defining the number and size of each fully connected layer.
        """
        if fc_layers is None:
            fc_layers = (100, 100)
        if isinstance(fc_layers, int):
            fc_layers = (fc_layers,)

        assert isinstance(gym_env_name, str), "passed gym_env_name not a string."
        assert gym_env_name != "", "gym environment name is empty."
        assert easyagents.env._is_registered_with_gym(gym_env_name), \
            f'"{gym_env_name}" is not the name of an environment registered with OpenAI gym.' + \
            'Consider using easyagents.env.register_with_gym to register your environment.'
        assert fc_layers is not None, "fc_layers not set"
        assert isinstance(fc_layers, tuple), "fc_layers not a tuple"
        assert fc_layers, "fc_layers must contain at least 1 int"
        for i in fc_layers:
            assert isinstance(i, int) and i >= 1, f'{i} is not a valid size for a hidden layer'

        self.original_env_name = gym_env_name
        self.gym_env_name = None
        self.fc_layers = fc_layers
        self.seed = seed

    def __str__(self):
        return f'fc_layers={self.fc_layers} seed={self.seed} gym_env_name={self.gym_env_name}'


class TrainContext(object):
    """Contains the current configuration of an agents train method like the number of iterations or the learning rate
        along with data gathered sofar during the training.

        The train loop proceeds roughly as follows:
            for i in num_iterations
                for e in num_episodes_per_iterations
                    play episode and record steps (while steps_in_episode < max_steps_per_episode and)
                train policy for num_epochs_per_iteration epochs
                if current_episode % num_iterations_between_eval == 0:
                    evaluate policy
                if training_done
                    break

        Hints:
        o TrainContext contains all the parameters needed to control the train loop.
        o TrainCallbacks get passed an instance of TrainContext and may modify its parameters, making a TrainCallback
            which dynamically adjust the learning rate possible.
        o Subclasses of TrainContext may contain additional Agent (but not backend) specific parameters.
        o TrainCallbacks may for example access training data gathered sofar for visualizations during training
        o see EasyAgent.train() for an outline of the train loop.

        Attributes:
            num_iterations: number of times the training is repeated (with additional data), unlimited if None
            num_episodes_per_iteration: number of episodes played per training iteration
            max_steps_per_episode: maximum number of steps per episode
            num_epochs_per_iteration: number of times the data collected for the current iteration
                is used to retrain the current policy
            learning_rate: the learning rate used in the next iteration's policy training (0,1]
            reward_discount_gamma: the factor by which a reward is discounted for each step (0,1]
            max_steps_in_buffer: size of the agents buffer in steps

            training_done: if true the train loop is terminated at the end of the current iteration
            iterations_done_in_training: the number of iterations completed so far (during training)
            episodes_done_in_iteration: the number of episodes completed in the current iteration
            episodes_done_in_training: the number of episodes completed over all iterations so far.
                The episodes played for evaluation are not included in this count.
            steps_done_in_training: the number of steps taken over all iterations so far
            steps_done_in_iteration: the number of steps taken in the current iteration

            loss: dict containing the loss for each iteration training. The dict is indexed by the current_episode.
            num_iterations_between_eval: number of training iterations before the current policy is evaluated.
                if 0 no evaluation is performed.
            num_episodes_per_eval: number of episodes played to estimate the average return and steps
            eval_rewards: dict containg the rewards statistics for each policy evaluation.
                Each entry contains the tuple (min, average, max) over the sum of rewardsover all episodes
                played for the current evaluation. The dict is indexed by the current_episode.
            eval_steps: dict containg the steps statistics for each policy evaluation.
                Each entry contains the tuple (min, average, max) over the number of step over all episodes
                played for the current evaluation. The dict is indexed by the current_episode.
    """

    def __init__(self):
        self.num_iterations: Optional[int] = None
        self.num_episodes_per_iteration: int = 10
        self.max_steps_per_episode: Optional = 1000
        self.num_epochs_per_iteration: int = 10
        self.num_iterations_between_eval: int = 50
        self.num_episodes_per_eval: int = 10
        self.learning_rate: float = 0.001
        self.reward_discount_gamma: float = 1
        self.max_steps_in_buffer: int = 100000

        self.training_done: bool
        self.iterations_done_in_training: int
        self.episodes_done_in_iteration: int
        self.episodes_done_in_training: int
        self.steps_done_in_training: int
        self.steps_done_in_iteration = 0
        self.loss: Dict[int, float]
        self.eval_rewards: Dict[int, Tuple[float, float, float]]
        self.eval_steps: Dict[int, Tuple[float, float, float]]
        self._reset()

    def __str__(self):
        return f'training_done={self.training_done} ' + \
                f'#iterations_done_in_training={self.iterations_done_in_training} ' + \
                f'#episodes_done_in_iteration={self.episodes_done_in_iteration} ' + \
                f'#steps_done_in_iteration={self.steps_done_in_iteration} ' + \
                f'#iterations={self.num_iterations} ' + \
                f'#episodes_per_iteration={self.num_episodes_per_iteration} ' + \
                f'#max_steps_per_episode={self.max_steps_per_episode} ' + \
                f'#epochs_per_iteration={self.num_epochs_per_iteration} ' + \
                f'#iterations_between_eval={self.num_iterations_between_eval} ' + \
                f'#episodes_per_eval={self.num_episodes_per_eval} ' + \
                f'#learning_rate={self.learning_rate} ' + \
                f'#reward_discount_gamma={self.reward_discount_gamma} ' + \
                f'#max_steps_in_buffer={self.max_steps_in_buffer} '


    def _validate(self):
        """Validates the consistency of all values, raising an exception if an inadmissible combination is detected."""
        assert self.num_iterations is None or self.num_iterations > 0, "num_iterations not admissible"
        assert self.num_episodes_per_iteration > 0, "num_episodes_per_iteration not admissible"
        assert self.max_steps_per_episode > 0, "max_steps_per_episode not admissible"
        assert self.num_epochs_per_iteration > 0, "num_epochs_per_iteration not admissible"
        assert self.num_iterations_between_eval > 0, "num_iterations_between_eval not admissible"
        assert self.num_episodes_per_eval > 0, "num_episodes_per_eval not admissible"
        assert 0 < self.learning_rate <= 1, "learning_rate not in interval (0,1]"
        assert 0 < self.reward_discount_gamma <= 1, "reward_discount_gamma not in interval (0,1]"

    def _reset(self):
        """Clears all values modified during a train() call."""
        self.training_done = False
        self.iterations_done_in_training = 0
        self.episodes_done_in_iteration = 0
        self.episodes_done_in_training = 0
        self.steps_done_in_training = 0
        self.steps_done_in_iteration = 0
        self.loss = dict()
        self.eval_rewards = dict()
        self.eval_steps = dict()


class ActorCriticTrainContext(TrainContext):
    """TrainContext for Actor-Critic type agents like Ppo or Sac.

    Attributes:
        actor_loss: loss observed during training of the actor network. dict is indexed by the current_episode.
        critic_loss: loss observed during training of the critic network. dict is indexed by the current_episode.
    """

    def __init__(self):
        super().__init__()
        self.actor_loss: Dict[int, float]
        self.critic_loss: Dict[int, float]

    def _reset(self):
        self.actor_loss = dict()
        self.critic_loss = dict()
        super()._reset()


class PlayContext(object):
    """Contains the current configuration of an agents play method like the number of episodes to play
        and the max number of steps per episode.

        The EasyAgent.play() method proceeds (roughly) as follow:

        for e in num_episodes
            play (while steps_done_in_episode < max_steps_per_episode)
            if playing_done
                break

        Attributes:
            num_episodes: number of episodes to play, unlimited if None
            max_steps_per_episode: maximum number of steps per episode, unlimited if None
            play_done: if true the play loop is terminated at the end of the current episode
            episodes_done: the number of episodes played (including the current episode).
            steps_done_in_episode: the number of steps taken in the current episode.
            steps_done: the number of steps played (over all episodes so far)

            actions: dict containing for each episode the actions taken in each step
            rewards: dict containing for each episode the rewards received in each step
            sum_of_rewards: dict containing for each episode the sum of rewards over all steps
            gym_env: the gym environment used to play
    """

    def __init__(self, train_context: Optional[TrainContext] = None):
        """
        Args:
             train_context: if set num_episodes, max_steps_per_episode and seed are set from train_context
        """
        self.num_episodes: Optional[int] = None
        self.max_steps_per_episode: Optional[int] = None

        if train_context is not None:
            self.num_episodes = train_context.num_episodes_per_eval
            self.max_steps_per_episode = train_context.max_steps_per_episode

        self.play_done: bool
        self.episodes_done: int
        self.steps_done_in_episode: int
        self.steps_done: int
        self.actions: Dict[int, List[object]]
        self.rewards: Dict[int, List[float]]
        self.sum_of_rewards: Dict[int, float]
        self.gym_env: Optional[gym.core.Env]
        self._reset()

    def __str__(self):
        return f'#episodes={self.num_episodes} ' +\
                f'max_steps_per_episode={self.max_steps_per_episode} '+\
                f'play_done={self.play_done} '+\
                f'episodes_done={self.episodes_done} '+\
                f'steps_done_in_episode={self.steps_done_in_episode} ' +\
                f'steps_done={self.steps_done} '

    def _validate(self):
        """Validates the consistency of all values, raising an exception if an inadmissible combination is detected."""
        assert self.num_episodes is None or self.num_episodes > 0, "num_episodes not admissible"
        assert self.max_steps_per_episode > 0, "max_steps_per_episode not admissible"

    def _reset(self):
        """Clears all values modified during a train() call."""
        self.play_done: bool = False
        self.episodes_done: int = 0
        self.steps_done_in_episode: int = 0
        self.steps_done: int = 0
        self.actions: Dict[int, List[object]] = dict()
        self.rewards: Dict[int, List[float]] = dict()
        self.sum_of_rewards: Dict[int, float] = dict()
        self.gym_env: Optional[gym.core.Env] = None


class AgentContext(object):
    """Collection of state and configuration settings for a EasyAgent instance.

    Attributes:
        model: model configuration including the name of the underlying gym_environment and the
            policy's neural network archtitecture.
        train: training configuration and current train state. None if not inside a train call.
        play: play / eval configuration and current state. None if not inside a play call (directly or
            due to a evaluation inside a train loop)
        gym: context for gym environment related calls.
        pyplot: the context containing the matplotlib.pyplot figure to plot to during training or playing
    """

    def __init__(self, model: ModelConfig):
        """
        Args:
            model: model configuration including the name of the underlying gym_environment and the
                policy's neural network archtitecture.
        """
        assert isinstance(model, ModelConfig), "model not set"

        self.model: ModelConfig = model
        self.train: Optional[TrainContext] = None
        self.play: Optional[PlayContext] = None
        self.gym: GymContext = GymContext()
        self.pyplot: PyPlotContext = PyPlotContext()

    def __str__(self):
        result = f'agent_context:'
        result += f'\napi   =[{self.gym}]'
        if self.train is not None:
            result += f'\ntrain =[{self.train}] '
        if self.play is not None:
            result += f'\nplay  =[{self.play}] '
        if self.pyplot is not None:
            result += f'\npyplot=[{self.pyplot}] '
        result += f'\nmodel =[{self.model}] '
        return result


class AgentCallback(ABC):
    """Base class for all callbacks monitoring the backend algorithms api calls or the api calls to the gym
        environment"""

    def on_api_log(self, agent_context: AgentContext, api_target: str, log_msg: str):
        """Logs a call to the api of the agents implementation library / framework."""
        pass

    def on_log(self, agent_context: AgentContext, log_msg: str):
        """Logs a general message"""
        pass

    def on_gym_init_begin(self, agent_context: AgentContext):
        """called when the monitored environment begins the instantiation of a new gym environment.

            Args:
                agent_context: api_context passed to calling agent
        """

    def on_gym_init_end(self, agent_context: AgentContext):
        """called when the monitored environment completed the instantiation of a new gym environment.

        Args:
            agent_context: api_context passed to calling agent
        """
        pass

    def on_gym_reset_begin(self, agent_context: AgentContext, **kwargs):
        """Before a call to gym.reset

            Args:
                agent_context: api_context passed to calling agent
                kwargs: the args to be passed to the underlying environment
        """

    def on_gym_reset_end(self, agent_context: AgentContext, reset_result: Tuple, **kwargs):
        """After a call to gym.reset was completed

        Args:
            agent_context: api_context passed to calling agent
            reset_result: object returned by gym.reset
            kwargs: args passed to gym.reset
        """
        pass

    def on_gym_step_begin(self, agent_context: AgentContext, action):
        """Before a call to gym.step

        Args:
            agent_context: api_context passed to calling agent
            action: the action to be passed to the underlying environment
        """
        pass

    def on_gym_step_end(self, agent_context: AgentContext, action, step_result: Tuple):
        """After a call to gym.step was completed

        Args:
            agent_context: api_context passed to calling agent
            action: the action to be passed to the underlying environment
            step_result: (observation,reward,done,info) tuple returned by gym.step
        """
        pass

    def on_play_episode_begin(self, agent_context: AgentContext):
        """Called once at the start of new episode to be played (during play or eval, but not during train). """

    def on_play_episode_end(self, agent_context: AgentContext):
        """Called once after an episode is done or stopped (during play or eval, but not during train)."""

    def on_play_begin(self, agent_context: AgentContext):
        """Called once at the entry of an agent.play() call (during play or eval, but not during train). """

    def on_play_end(self, agent_context: AgentContext):
        """Called once before exiting an agent.play() call (during play or eval, but not during train)"""

    def on_play_step_begin(self, agent_context: AgentContext, action):
        """Called once before a new step is taken in the current episode (during play or eval, but not during train).

            Args:
                 agent_context: the context describing the agents current configuration
                 action: the action to be passed to the upcoming gym_env.step call
        """

    def on_play_step_end(self, agent_context: AgentContext, action, step_result: Tuple):
        """Called once after a step is completed in the current episode (during play or eval, but not during train)."""

    def on_train_begin(self, agent_context: AgentContext):
        """Called once at the entry of an agent.train() call. """

    def on_train_end(self, agent_context: AgentContext):
        """Called once before exiting an agent.train() call"""

    def on_train_iteration_begin(self, agent_context: AgentContext):
        """Called once at the start of a new iteration. """

    def on_train_iteration_end(self, agent_context: AgentContext):
        """Called once after the current iteration is completed"""
