from abc import ABC, abstractmethod
import base64
import os
import tempfile
from logging import INFO, getLogger
from sys import maxsize, float_info
from typing import List, Tuple

import gym
import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import Figure

from easyagents.config import Logging
from easyagents.config import Training
from easyagents.easyenv import _EasyEnv
from easyagents.easyenv import _register

# check if we are running in Jupyter, if so interactive plotting must be handled differently
# (in order to get plot updates during training)
_is_jupyter_active = False
try:
    # noinspection PyUnresolvedReferences
    from IPython import get_ipython
    # noinspection PyUnresolvedReferences
    from IPython.display import display, clear_output
    # noinspection PyUnresolvedReferences
    from IPython.display import HTML

    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell':
        _is_jupyter_active = True
    else:
        # noinspection PyPackageRequirements
        import google.colab

        _is_jupyter_active = True
except ImportError:
    pass

# download mp4 rendering
imageio.plugins.ffmpeg.download()


class EasyAgent(ABC):
    """ Abstract base class for all easy reinforcment learning agents.

        Args:
        gym_env_name            : the name of the registered gym environment to use, eg 'CartPole-v0'
        fc_layers               : int tuple defining the number and size of each fully connected layer
        training                : instance of config.Training to configure the #episodes used for training.
        learning_rate           : value in (0,1]. Factor by which the impact on the policy update is reduced
                                  for each training step. The same learning rate is used for the value and
                                  the policy network.
        reward_discount_gamma   : value in (0,1]. Factor by which a future reward is discounted for each step.
        logging                 : instance of config.Logging to configure the logging behaviour 
    """

    def __init__(self,
                 gym_env_name: str,
                 training: Training = None,
                 fc_layers=None,
                 learning_rate: float = 0.001,
                 reward_discount_gamma: float = 1,
                 logging: Logging = None):
        if fc_layers is None:
            fc_layers = (75, 75)
        if training is None:
            training = Training()
        if logging is None:
            logging = Logging()

        assert isinstance(gym_env_name, str), "passed gym_env_name not a string."
        assert gym_env_name != "", "gym environment name is empty."
        assert fc_layers is not None, "fc_layers not set"
        assert isinstance(training, Training), "training not an instance of easyagents.config.Training"
        assert learning_rate > 0, "learning_rate must be in (0,1]"
        assert learning_rate <= 1, "learning_rate must be in (0,1]"
        assert reward_discount_gamma > 0, "reward_discount_gamma must be in (0,1]"
        assert reward_discount_gamma <= 1, "reward_discount_gamma must be in (0,1]"
        assert isinstance(logging, Logging), "logging not an instance of easyagents.config.Logging"

        self._gym_env_name = gym_env_name
        self._training = training
        self.fc_layers = fc_layers
        self._learning_rate = learning_rate
        self._reward_discount_gamma = reward_discount_gamma
        self._logging = logging
        self.training_average_rewards = []
        self.training_average_steps = []
        self.training_losses = []

        self._train_figure = None
        self._train_is_jupyter_display_figure = False
        self._train_render_rgb_array = np.array([])

        self._log = getLogger(name=__name__)
        self._log.setLevel(INFO)
        self._log_minimal(f'{self}')
        self._log_minimal(f'Training {self._training}')

        self._gym_env_name = _register(gym_env_name=self._gym_env_name,
                                       log_api=self._logging.log_gym_api,
                                       log_steps=self._logging.log_gym_api_steps,
                                       log_reset=self._logging.log_gym_api_reset)
        return

    def __str__(self):
        """ yields a human readable representation of the agents/algorithms current configuration
        """
        result = f'{type(self).__name__} on {self._gym_env_name} [fc_layers={self.fc_layers}, ' + \
                 f'learning_rate={self._learning_rate}'
        if self._reward_discount_gamma < 1:
            result += f', reward_discount_gamma={self._reward_discount_gamma}'
        result += ']'
        return result

    def _log_agent(self, msg):
        if self._logging.log_agent:
            self._log.info(msg)
        return

    def _log_minimal(self, msg):
        if self._logging.log_minimal or self._logging.log_agent:
            self._log.info(msg)
        return

    def play_episode(self, max_steps: int = None, callback: object = None) -> object:
        """ Plays a full episode using the previously trained policy, yielding
            the sum of rewards, the totale number of steps taken over the episode.

            Args:
            max_steps   : if the episode is not done after max_steps it is aborted.
                          if None the episode ends only when the environments step yields done.
            callback    : callback(action,state,reward,step,done,info) is called after each step.
                          if the callback yields True, the episode is aborted.
            :returns rewards,steps
        """
        return self._play_episode(max_steps=max_steps, callback=callback)

    def _play_episode(self, max_steps: int, callback) -> (float, int, bool):
        """ Plays a full episode using the previously trained policy, yielding
            the sum of rewards, the totale number of steps taken over the episode.

            Args:
            max_steps   : if the episode is not done after max_steps it is aborted.
                          if None the episode ends only when the environments step yields done.
            callback    : callback(action,state,reward,step,done,info) is called after each step.
                          if the callback yields True, the episode is aborted.
            :returns rewards,steps
        """
        return 0.0, 0

    def plot_episodes(self, ylim: List[Tuple[float, float]] = None, scale: List[str] = None):
        """ creates and displays a figure with 3 subplots representing some training statistics:
                o loss (training loss after each model update)
                o average sum of rewards/episode (averaged over Logging.num_eval_episodes episodes)
                o average num of steps/ episode (averaged over Logging.num_eval_episodes episodes)

            All list arguments - if not None - must contain exactly 1 value for each of the plots.

        :param ylim: ylim for each plot, eg [(-100,100),(0,10),(0,50)]. By default no limits.
        :param scale: scale for each plot, eg ['log,'linear','linear']. By default linear for all plots
        :return: figure, axes
        """
        figure = self._plot_episodes(ylim=ylim, scale=scale, is_jupyter_display_figure=False)
        return figure

    def _plot_episodes(self, ylim: List[Tuple[float, float]] = None, scale: List[str] = None,
                       figure: Figure = None, is_jupyter_display_figure: bool = False,
                       rgb_array: np.ndarray = None):
        """ Draws a figure with 3 subplots. If rgb_array is not None then an additional
            subplot with an image of the rgb_array is added.

            if is_jupyter_display_figure is set, then display(figure) is called if we are running
            inside a jupyter notebook. Hereby an initial doubled figure output is avoided.
        """
        assert ylim is None or len(ylim) == 3, "ylim must contain an (float,float) for each of the 3 plots."
        assert scale is None or len(scale) == 3, "scale must contain an 'linear' or 'log' for each of the 3 plots."

        if figure is None:
            x_set, y_set = 17, 5
            figure = plt.figure("EasyAgents", figsize=(x_set, y_set))
            for axis in figure.axes:
                figure.delaxes(axis)
            num_subplots = 3 if rgb_array is None else 4
            axes = [figure.add_subplot(1, num_subplots, i + 1) for i in range(num_subplots)]
            if rgb_array is not None:
                figure.tight_layout(w_pad=3)
        else:
            axes = figure.axes
            axeslen = len(axes)
            assert axeslen >= 3, f'figure contains {axeslen} axes, but must contain at least 3 axes'
            assert axeslen <= 4, f'figure contains {axeslen} axes, but may contain at most 4 axes'
        if scale is None:
            scale = ['log', 'linear', 'linear']
        if ylim is None:
            ylim = [None] * 3

        # make sure the plotting takes place on our figure
        if _is_jupyter_active:
            clear_output(wait=True)
            for ax in axes:
                ax.cla()
        else:
            plt.figure(figure.number)
            plt.cla()

        # plot
        # draw rgb image if available
        offset = 0
        if rgb_array is not None:
            ax = axes[0]
            ax.imshow(rgb_array)
            ax.set_xlabel("'done state' of last evaluation episode")
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            axes_color = 'grey'
            for spin in ax.spines:
                ax.spines[spin].set_color(axes_color)
            offset = 1
        # plot statistics (loss, rewards, steps) with same x-axes
        episodes_per_value = self._training.num_episodes_per_iteration
        xlim = episodes_per_value * (len(self.training_losses) - 1)
        xlim = 1 if xlim <= 1 else xlim
        self._subplot(axes=axes[0 + offset], yvalues=self.training_losses, episodes_per_value=episodes_per_value,
                      ylabel='loss', ylim=ylim[0], scale=scale[0], xlim=xlim, color='indigo')

        episodes_per_value = self._training.num_episodes_per_iteration * self._training.num_iterations_between_eval
        self._subplot(axes=axes[1 + offset], yvalues=self.training_average_rewards,
                      episodes_per_value=episodes_per_value,
                      ylabel='sum of rewards', ylim=ylim[1], scale=scale[1], xlim=xlim, color='g')
        self._subplot(axes=axes[2 + offset], yvalues=self.training_average_steps, episodes_per_value=episodes_per_value,
                      ylabel='steps', ylim=ylim[2], scale=scale[2], xlim=xlim, color='b')

        # make sure the plots are presented to the user
        if _is_jupyter_active and is_jupyter_display_figure:
            display(figure)
        plt.pause(0.01)
        return figure

    def _subplot(self, axes: plt.Axes, yvalues, episodes_per_value: int,
                 ylabel: str, ylim, scale: str, xlim: int, color: str):
        """ plot yvalues on axes.
            if yvalues is of the form [y1, y2,...] then a simple line is dran in color
            if yvalues is of the form [(min,y,max),...] then a min/max area around y is drawn as well
        """
        value_count = len(yvalues)
        steps = range(0, value_count * episodes_per_value, episodes_per_value)
        figure = axes.figure

        # under unittest the current figure seems not to be available anymore
        if not _is_jupyter_active:
            plt.figure(figure.number)
            if figure == plt.gcf():
                plt.sca(axes)

        # setup subplot (axes, labels, colors)
        axes_color = 'grey'
        axes.set_xlabel('episodes')
        axes.set_ylabel(ylabel)
        axes.set_xlim(0, xlim)
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        axes.spines['bottom'].set_color(axes_color)
        axes.spines['left'].set_color(axes_color)
        axes.grid(color=axes_color, linestyle='-', linewidth=0.25, alpha=0.5)
        if ylim is not None:
            axes.set_ylim(ylim)
        axes.set_yscale(scale)

        # extract min / max and y values if yvalues is of the form [(min,y,max),...)
        yminvalues = None
        ymaxvalues = None
        if len(yvalues) > 0 and isinstance(yvalues[0], tuple):
            ymaxvalues = [t[2] for t in yvalues]
            yminvalues = [t[0] for t in yvalues]
            yvalues = [t[1] for t in yvalues]

        # plot values
        fill_alpha = 0.1
        if _is_jupyter_active:
            if yminvalues is not None:
                axes.fill_between(steps, yminvalues, ymaxvalues, color=color, alpha=fill_alpha)
            axes.plot(steps, yvalues, color=color)
        else:
            if figure == plt.gcf():
                plt.sca(axes)
            if yminvalues is not None:
                plt.fill_between(steps, yminvalues, ymaxvalues, color=color, alpha=fill_alpha)
            plt.plot(steps, yvalues, color=color)
            plt.pause(0.01)

    def render_episodes(self, num_episodes: int = 1, mode='human'):
        """ plays num_episodes, calling and environment.render after each step.

            gym_env.render(mode) is called (which should render on the current display or terminal)

            Args:
            num_episodes    : the number of episodes to render
            mode            : the mode argument passed to render (typically 'human' or 'ansi')
                              If 'ansi' then the rendered string is printed to the console.
        """
        assert num_episodes >= 0, "num_episodes must be >= 0"
        for _ in range(num_episodes):
            if mode == 'ansi':
                self.play_episode(
                    callback=lambda gym_env, action, state, reward, step, done, info:
                    print(gym_env.render(mode=mode)))
            else:
                self.play_episode(
                    callback=lambda gym_env, action, state, reward, step, done, info: gym_env.render(mode=mode))

    def render_episodes_to_jupyter(self,
                                   num_episodes: int = 10,
                                   fps: int = 20,
                                   width: int = 640,
                                   height: int = 480,
                                   mode='rgb_array'):
        """ renders all steps in num_episodes as a mp4 movie and displays it in the surrounding jupyter notebook.

            The gym_env.render(mode='rgb_array') must yield an numpy.ndarray representing rgb values, 
            otherwise an exception is thrown.

            Args:
            num_episodes    : the number of episodes to render
            fps             : frames per second, each frame contains the rendering of a single step
            height          : height iin pixels of the HTML rendered episodes
            width           : width in pixels of the HTML rendered episodes

            Note:
            o code adapted from:
            https://colab.research.google.com/github/tensorflow/agents/blob/master/tf_agents/colabs/1_dqn_tutorial.ipynb
        """
        assert num_episodes >= 0, "num_episodes must be >= 0"
        assert height >= 1, "height must be >= 1"
        assert width >= 1, "width must be >= 1"
        assert _is_jupyter_active, "must be running inside a jupyter notebook"

        filepath = self.render_episodes_to_mp4(num_episodes=num_episodes, fps=fps, mode=mode)
        with open(filepath, 'rb') as f:
            video = f.read()
            b64 = base64.b64encode(video)
        os.remove(filepath)

        result = '''
        <video width="{0}" height="{1}" controls>
            <source src="data:video/mp4;base64,{2}" type="video/mp4">
        Your browser does not support the video tag.
        </video>'''.format(width, height, b64.decode())
        result = HTML(result)
        # noinspection PyTypeChecker
        display(result)

    def render_episodes_to_mp4(self, num_episodes: int = 10, filepath: str = None, fps: int = 20,
                               mode='rgb_array') -> str:
        """ renders all steps in num_episodes as a mp4 movie and stores it in filename.
            Returns the path to the written file.

            The gym_env.render(mode='rgb_array') must yield an numpy.ndarray representing rgb values, 
            otherwise an exception is thrown.

            Args:
                num_episodes: the number of episodes to render
                filepath: the path to which the movie is written to. If None a temp filepath is generated.
                fps: frames per second
                mode: the mode passed to the env's render function

            Note:
                code adapted from:
                https://colab.research.google.com/github/tensorflow/agents/blob/master/tf_agents/colabs/1_dqn_tutorial.ipynb
        """
        assert num_episodes >= 0, "num_episodes must be >= 0"

        if filepath is None:
            filepath = self._gym_env_name
            if filepath.startswith(_EasyEnv._NAME_PREFIX):
                filepath = filepath[len(_EasyEnv._NAME_PREFIX):]
            filepath = os.path.join(tempfile.gettempdir(),
                                    next(tempfile._get_candidate_names()) + "_" + filepath + ".mp4")
        with imageio.get_writer(filepath, fps=fps) as video:
            for _ in range(num_episodes):
                self.play_episode(
                    callback=lambda gym_env, action, state, reward, step, done, info:
                    video.append_data(self._render_to_rgb_array(gym_env, mode=mode)))
        return filepath

    def _render_to_rgb_array(self, gym_env: gym.Env, mode: str = 'rgb_array') -> np.ndarray:
        """ calls gym_env.render(mode) and validates the return value to be a numpy rgb array
            throws an exception if not an rgb array

            Returns:
                numpy rgb array
        """
        result = gym_env.render(mode=mode)

        assert result is not None, f'gym_env.render(mode={mode}) yielded None'
        assert isinstance(result, np.ndarray), f'gym_env.render(mode={mode}) did not yield a numpy.ndarray.'
        assert result.min() >= 0, f'gym_env.render(mode={mode}) contains negative values => not an rgb array'
        assert result.max() <= 255, f'gym_env.render(mode={mode}) contains values > 255 => not an rgb array'
        assert len(result.shape) == 3, f'gym_env.render(mode={mode}) shape is not of the form (x,y,n)'
        assert result.shape[2] == 3 or result.shape[2] == 4, \
            f'gym_env.render(mode={mode}) shape is not of the form (x,y,3|4)'
        return result

    def train(self):
        """ trains a policy using the gym_env.
            Sets training_losses and training_average_returns, depending on the training scheme
            defined in Training configuration.
        """
        self.training_average_rewards = []
        self.training_average_steps = []
        self.training_losses = [0]
        self._train_figure = None
        self._train_is_jupyter_display_figure = False
        self._train_render_rgb_array = np.array([])
        self._train()

    @abstractmethod
    def _train(self):
        """ overriden by the implementing agent and called by train()
        """
        return

    # noinspection PyShadowingNames
    def _train_eval_rewards_and_steps(self):
        """ computes the expected sum of rewards and the expected step count for the previously trained policy.
            and adds them to the training logs.
            If the gym_env supports rgb_array rendering the last game_state is rendered and stored.

            Note:
            The evaluation is performed on a instance of gym_env_name.
        """

        def render_to_rgb_array(is_render: bool, gym_env: _EasyEnv):
            if is_render and self._train_render_rgb_array is not None:
                # noinspection PyBroadException
                try:
                    self._train_render_rgb_array = None
                    self._train_render_rgb_array = self._render_to_rgb_array(gym_env)
                except BaseException:
                    pass
            return

        self._log_agent(f'current policy       : evaluating... [{self._training.num_eval_episodes} episodes]')
        max_reward = float_info.min
        min_reward = float_info.max
        sum_rewards = 0.0
        max_steps = 0
        min_steps = maxsize
        sum_steps = 0
        num_episodes = self._training.num_eval_episodes
        max_eval_steps = self._training.max_steps_per_episode
        for i in range(num_episodes):
            (reward, steps) = self.play_episode(
                max_steps=max_eval_steps,
                callback=lambda gym_env, action, state, reward, step, done, info:
                render_to_rgb_array(self._logging.plots and
                                    (i == (num_episodes - 1) and (done or step == max_eval_steps)),
                                    gym_env))
            max_reward = max(max_reward, reward)
            min_reward = min(min_reward, reward)
            max_steps = max(max_steps, steps)
            min_steps = min(min_steps, steps)
            sum_rewards += reward
            sum_steps += steps
        avg_rewards: float = sum_rewards / num_episodes
        avg_steps: float = sum_steps / num_episodes
        self._log_minimal(
            f'current policy       : avg_reward={float(avg_rewards):.3f}, avg_steps={float(avg_steps):.3f}')
        self.training_average_rewards.append((min_reward, avg_rewards, max_reward))
        self.training_average_steps.append((min_steps, avg_steps, max_steps))

    def _train_iteration_completed(self, iteration: int, total_loss: float = 0):
        """called by the implementing agent in _train after each completed training iteration.

        performs houskeeping to update the visualization and statistics

        if called with iteration == 0 an initial evaluation of an untrained policy is performed

        Args:
            iteration: the count of the completed iteration (starting at 1)
            total_loss: the loss for this training iteration
        """
        assert iteration >= 0, "passed iteration < 0"

        if iteration > 0:
            if len(self.training_average_rewards) == 0:
                # overriding agent failed to to do a call with iteration==0: compensate.
                self.training_average_rewards = [0]
                self.training_average_steps = [0]
            msg = f'iteration {iteration:4} of {self._training.num_iterations:<4}:'
            self.training_losses.append(float(total_loss))
            self._log_minimal(f'{msg} completed tf_agent.train(...) = {total_loss.numpy():>8.3f} [loss]')

        if iteration % self._training.num_iterations_between_eval == 0:
            self._train_eval_rewards_and_steps()
        if self._logging.plots:
            self._train_figure = self._plot_episodes(figure=self._train_figure,
                                                     is_jupyter_display_figure=self._train_is_jupyter_display_figure,
                                                     rgb_array=self._train_render_rgb_array)
            self._train_is_jupyter_display_figure = True
        return


class AbstractAgent(EasyAgent):
    """ Base class for all Agent implementations.

        Note: Do not derive from EasyAgent directly, always derived from AbstractAgent (or a subclass thereof) instead.
    """

    @abstractmethod
    def _play_episode(self, max_steps: int, callback) -> (float, int, bool):
        """ Plays a full episode using the previously trained policy, yielding
            the sum of rewards, the totale number of steps taken over the episode.

            Args:
                max_steps: if the episode is not done after max_steps it is aborted.
                    if None the episode ends only when the environments step yields done.
                callback: callback(action,state,reward,step,done,info) is called after each step.
                    if the callback yields True, the episode is aborted.

            Returns:
                rewards,steps
        """
        return super()._play_episode(max_steps, callback)

    @abstractmethod
    def _train(self):
        """ overriden by the implementing agent and called by train()

            Note:
            o Call _train_iteration_completed(0) after the instantiation of the model is completed
              (after this call, calling _play_episode is admissible).
            o Call _train_iteration_completed(iteration) after each completed iteration.
        """
        super().train()
