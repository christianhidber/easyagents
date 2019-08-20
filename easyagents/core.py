"""This module contains the core datastructures shared between fronten and backend like the definition
    of all callbacks and agent configurations.
"""

from abc import ABC
from typing import Optional, Dict, Tuple, List

from easyagents.env import _is_registered_with_gym


class ApiContext(object):
    """Contains the context of the backend or gym api call

        Attributes:
            gym_env: the gym environment for gym api calls, None otherwise
    """
    def __init__(self):
        self.gym_env = None


class ApiCallback(ABC):
    """Base class for all callbacks monitoring the backend algorithms api calls or the api calls to the gym
        environment"""

    def on_backend_call_begin(self, call_name: str, play_context: ApiContext):
        """Before a call into a backend implementation."""

    def on_backend_call_end(self, call_name: str, play_context: ApiContext):
        """After a call into a backend implementation was completed."""

    def on_gym_init_begin(self, play_context: ApiContext):
        """Before a call to gym.init"""

    def on_gym_init_end(self, play_context: ApiContext):
        """After a call to gym.init was completed"""

    def on_gym_step_begin(self, play_context: ApiContext):
        """Before a call to gym.step"""

    def on_gym_step_end(self, play_context: ApiContext):
        """After a call to gym.step was completed"""

    def on_gym_reset_begin(self, play_context: ApiContext):
        """Before a call to gym.reset"""

    def on_gym_reset_end(self, play_context: ApiContext):
        """After a call to gym.reset was completed"""


class ModelConfig(object):
    """The model configurations, containing the name of the gym environment and the neural network architecture.

    Args:
        gym_env_name: the name of the registered gym environment to use, eg 'CartPole-v0'
        fc_layers: int tuple defining the number and size of each fully connected layer.
    """

    def __init__(self, gym_env_name: str, fc_layers: Optional[Tuple[int, ...]] = None):
        if fc_layers is None:
            fc_layers = (100, 100)
        if isinstance(fc_layers, int):
            fc_layers = (fc_layers,)

        assert isinstance(gym_env_name, str), "passed gym_env_name not a string."
        assert gym_env_name != "", "gym environment name is empty."
        assert _is_registered_with_gym(gym_env_name), \
            f'"{gym_env_name}" is not the name of an environment registered with OpenAI gym.' + \
            'Consider using easyagents.env.register_with_gym to register your environment.'
        assert fc_layers is not None, "fc_layers not set"
        assert isinstance(fc_layers, tuple), "fc_layers not a tuple"
        assert fc_layers, "fc_layers must contain at least 1 int"
        for i in fc_layers:
            assert isinstance(i, int) and i >= 1, f'{i} is not a valid size for a hidden layer'

        self.gym_env_name = gym_env_name
        self.fc_layers = fc_layers


class PlayContext(object):
    """Contains the current configuration of an agents play method like the number of episodes to play
        and the max number of steps per episode.

        The EasyAgent.play() method proceeds (roughly) as follow:

        for e in num_episodes
            play (while current_steps_in_episode < max_steps_per_episode)
            if playing_done
                break

        Attributes:
            num_episodes: number of episodes to play, unlimited if None
            max_steps_per_episode: maximum number of steps per episode, unlimited if None
            play_done: if true the play loop is terminated at the end of the current episode
            current_episode: the number of episodes played (including the current episode).
            current_steps_in_episode: the number of steps taken in the current episode.
            current_steps: the number of steps played (over all episodes so far)
            rewards: dict containg for each episode the reward received in each step
            gym_env: the gym environment instance used to play
    """

    def __init__(self):
        self.gym_env = None
        self.num_episodes: Optional[int] = None
        self.max_steps_per_episode: Optional[int] = None
        self.play_done: bool = False
        self.current_episode: int = 0
        self.current_steps_in_episode: int = 0
        self.current_steps: int = 0
        self.rewards: Dict[int, List[float]] = dict()


class PlayCallback(ABC):
    """Base class for all callbacks monitoring the evaluation or the play of an agent"""

    def on_episode_begin(self, play_context: PlayContext):
        """Called once at the start of new episode to be played. """

    def on_episode_end(self, play_context: PlayContext):
        """Called once after an episode is done or stopped."""

    def on_play_begin(self, play_context: PlayContext):
        """Called once at the entry of an agent.play() call. """

    def on_play_end(self, play_context: PlayContext):
        """Called once before exiting an agent.play() call"""

    def on_step_begin(self, play_context: PlayContext):
        """Called once before a new step is taken in the current episode. """

    def on_step_end(self, play_context: PlayContext):
        """Called once after a step is completed in the current episode."""


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
            num_episodes_per_iteration: number of episodes played per training iteration, unlimited if None
            max_steps_per_episode: maximum number of steps per episode, unlimited if None
            num_epochs_per_iteration: number of times the data collected for the current iteration
                is used to retrain the current policy
            learning_rate: the learning rate used in the next iteration's policy training (0,1]

            train_done: if true the train loop is terminated at the end of the current iteration
            current_iteration: the index of the current training iteration, starting at 0.
            current_episode_in_iteration: the index of the current episode in the current iteration, starting at 0.
            current_episode: the number of episodes played during training (including the current episode).
                The episodes played for evaluation are not included in this count.

            loss: dict containing the loss for each iteration training. The dict is indexed by the current_episode.
            num_iterations_between_eval: number of training iterations before the current policy is evaluated
            num_episodes_per_eval: number of episodes played to estimate the average return and steps
            eval_average_rewards: dict containg the rewards statistics for each policy evaluation.
                Each entry contains the tuple (min, average, max) over the sum of rewardsover all episodes
                played for the current evaluation. The dict is indexed by the current_episode.
            eval_average_steps: dict containg the steps statistics for each policy evaluation.
                Each entry contains the tuple (min, average, max) over the number of step over all episodes
                played for the current evaluation. The dict is indexed by the current_episode.
            gym_env: the gym environment instance used to play
    """

    def __init__(self):
        self.gym_env = None
        self.num_iterations: Optional[int] = None
        self.num_episodes_per_iteration: Optional[int] = None
        self.max_steps_per_episode: Optional[int] = None
        self.num_epochs_per_iteration: Optional[int] = None
        self.num_iterations_between_eval: int = 100
        self.num_episodes_per_eval: int = 10
        self.learning_rate: float = 1
        self.train_done: bool = False
        self.current_iteration: int = 0
        self.current_episode_in_iteration: int = 0
        self.current_episode: int = 0
        self.loss: Dict[int, float] = dict()
        self.eval_average_rewards: Dict[int, Tuple[float, float, float]] = dict()
        self.eval_average_steps: Dict[int, Tuple[float, float, float]] = dict()


class TrainCallback(ABC):
    """Base class for all callbacks controlling / monitoring the training loop of an agent"""

    def on_train_begin(self, train_context: TrainContext):
        """Called once at the entry of an agent.train() call. """

    def on_train_end(self, train_context: TrainContext):
        """Called once before exiting an agent.train() call"""

    def on_iteration_begin(self, train_context: TrainContext):
        """Called once at the start of a new iteration. """

    def on_iteration_end(self, train_context: TrainContext):
        """Called once after the current iteration is completed"""
