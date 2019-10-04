from typing import Optional, List, Tuple, Union, Dict

import easyagents.core as core
import base64
import matplotlib.pyplot as plt
import numpy as np
import imageio
import math
import gym
import tempfile
import os.path
import datetime
import time

# download mp4 rendering
imageio.plugins.ffmpeg.download()

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


class _PreProcess(core._PreProcessCallback):
    """Initializes the matplotlib agent_context.pyplot.figure"""

    def _setup(self, agent_context: core.AgentContext):
        # create figure / remove all existing axes from previous calls to train/play
        pyc = agent_context.pyplot
        pyc.is_jupyter_active = _is_jupyter_active

        if pyc.figure is None:
            pyc.figure = plt.figure("_EasyAgents", figsize=pyc.figsize)
        for ax in pyc.figure.axes:
            pyc.figure.delaxes(ax)

    def on_play_begin(self, agent_context: core.AgentContext):
        # play_begin is also called at the start of a policy evaluation
        if agent_context.is_play:
            self._setup(agent_context=agent_context)

    def on_train_begin(self, agent_context: core.AgentContext):
        self._setup(agent_context=agent_context)


class _PostProcess(core._PostProcessCallback):
    """Redraws the plots int the matplotlib agent_context.figure.

        In jupyter the plots are only refreshed once per second.
    """

    def __init__(self):
        self._call_jupyter_display: bool
        self._reset()

    def _reset(self):
        self._call_jupyter_display = False

    def _display(self, agent_context: core.AgentContext, wait=True):
        """Fixes the layout of multiple subplots and refreshs the display."""
        pyc = agent_context.pyplot
        count = len(pyc.figure.axes)
        if count > 0:
            rows = math.ceil(count / pyc.max_columns)
            columns = math.ceil(count / rows)
            for i in range(count):
                pyc.figure.axes[i].change_geometry(rows, columns, i + 1)
            pyc.figure.tight_layout()

            if pyc.is_jupyter_active:
                clear_output(wait=True)
                if self._call_jupyter_display:
                    # noinspection PyTypeChecker
                    display(pyc.figure)
                self._call_jupyter_display = True
            else:
                plt.pause(0.01)

    def on_play_begin(self, agent_context: core.AgentContext):
        if agent_context.is_play:
            self._reset()

    def on_train_begin(self, agent_context: core.AgentContext):
        self._reset()

    def on_play_episode_end(self, agent_context: core.AgentContext):
        if agent_context._is_plot_ready(core.PlotType.PLAY_EPISODE):
            self._display(agent_context, wait=False)

    def on_play_step_end(self, agent_context: core.AgentContext, action, step_result: Tuple):
        if agent_context._is_plot_ready(core.PlotType.PLAY_STEP):
            self._display(agent_context, wait=False)

    def on_play_end(self, agent_context: core.AgentContext):
        if agent_context._is_plot_ready(core.PlotType.TRAIN_EVAL):
            self._display(agent_context, wait=False)
        if agent_context.is_play:
            self._display(agent_context)

    def on_train_end(self, agent_context: core.AgentContext):
        self._display(agent_context)
        self._reset()
        self._display(agent_context)

    def on_train_iteration_end(self, agent_context: core.AgentContext):
        if agent_context._is_plot_ready(core.PlotType.TRAIN_ITERATION):
            self._display(agent_context, wait=False)


# noinspection DuplicatedCode
class _PlotCallback(core.AgentCallback):
    """Base class of plyplot callbacks generating a plot after a trained iteration or an episode played.

        Attributes:
            axes: the subplot to plot onto
    """

    def __init__(self, plot_type: core.PlotType):
        """Base class of plyplot callbacks generating a plot after a trained iteration or an episode played.

            Args:
                plot_type: point in time when the plot is updated
        """
        self.axes = None
        self.axes_color: str = 'grey'
        self._plot_type: core.PlotType = plot_type

    def _create_subplot(self, agent_context: core.AgentContext):
        if self.axes is None:
            pyc = agent_context.pyplot
            pyc._created_subplots = pyc._created_subplots | self._plot_type
            count = len(pyc.figure.axes) + 1
            rows = math.ceil(count / pyc.max_columns)
            columns = math.ceil(count / rows)
            self.axes = pyc.figure.add_subplot(rows, columns, count)
            self.plot_axes(xlim=(0, 1), ylabel='', xlabel='')

    def _refresh_subplot(self, agent_context: core.AgentContext, plot_type: core.PlotType):
        """Sets this axes active and calls plot if this plot callback is registered on at least 1 plot
            out of plot_type."""
        assert self.axes is not None
        plot_type = plot_type & self._plot_type
        if agent_context._is_plot_ready(plot_type):
            pyc = agent_context.pyplot
            if not pyc.is_jupyter_active:
                plt.figure(pyc.figure.number)
                if plt.gcf() is pyc.figure:
                    plt.sca(self.axes)
            self.plot(agent_context, plot_type)

    def _is_nan(self, values: Optional[List[float]]):
        """yields true if all values are equal to nan. yields false if values is None or empty."""
        result = False
        if values and all(isinstance(v,float) for v in values):
            result = all(math.isnan(v) for v in values)
        return result

    def clear_plot(self, agent_context: core.AgentContext):
        """Clears the axes for this plot. Should be called by self.plot before replotting an axes."""
        assert self.axes is not None
        pyc = agent_context.pyplot
        if pyc.is_jupyter_active:
            self.axes.cla()
        else:
            plt.cla()

    def on_play_begin(self, agent_context: core.AgentContext):
        if (self._plot_type & (core.PlotType.PLAY_EPISODE | core.PlotType.PLAY_STEP)) != core.PlotType.NONE:
            self._create_subplot(agent_context)

    def on_play_end(self, agent_context: core.AgentContext):
        self._refresh_subplot(agent_context, core.PlotType.TRAIN_EVAL)

    def on_play_episode_end(self, agent_context: core.AgentContext):
        self._refresh_subplot(agent_context, core.PlotType.PLAY_EPISODE)

    def on_play_step_end(self, agent_context: core.AgentContext, action, step_result: Tuple):
        self._refresh_subplot(agent_context, core.PlotType.PLAY_STEP)

    def on_train_begin(self, agent_context: core.AgentContext):
        if (self._plot_type & (core.PlotType.TRAIN_EVAL | core.PlotType.TRAIN_ITERATION)) != core.PlotType.NONE:
            self._create_subplot(agent_context)

    def on_train_iteration_end(self, agent_context: core.AgentContext):
        self._refresh_subplot(agent_context, core.PlotType.TRAIN_ITERATION)

    def plot(self, agent_context: core.AgentContext, plot_type: core.PlotType):
        """Plots a graph on the self.axes object.

        Args:
            agent_context: the context containing the data to plot
            plot_type: the PlotType requesting this plot call.
        """
        pass

    def plot_axes(self, xlabel: str, ylabel: str, xlim: Tuple[float, float] = None,
                  yscale: str = 'linear', ylim: Optional[Tuple[float, float]] = None):
        """Draws the x- and y-axes.

        Attributes:
            xlim: None or (min,max) for the x-axes
            xlabel: label of the x-axes
            ylim: None or (min,max) for the y-axes (or None)
            ylabel: label of the y-axes
            yscale: scale of the y-axes ('linear', 'log', ...)
        """
        assert xlim is None or xlim[0] <= xlim[1]
        assert ylim is None or ylim[0] <= ylim[1]
        assert xlabel is not None
        assert ylabel is not None

        # setup subplot (axes, labels, colors)
        axes_color = self.axes_color
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.spines['top'].set_visible(False)
        self.axes.spines['right'].set_visible(False)
        self.axes.spines['bottom'].set_color(axes_color)
        self.axes.spines['left'].set_color(axes_color)
        self.axes.grid(color=axes_color, linestyle='-', linewidth=0.25, alpha=0.5)
        if xlim is not None:
            self.axes.set_xlim(xlim)
        if ylim is not None:
            self.axes.set_ylim(ylim)
        self.axes.set_yscale(yscale)

    def plot_text(self, text: str):
        if text:
            ax = self.axes
            ax.text(0.5, 0.5, text, horizontalalignment='center', verticalalignment='center',
                    color='blue', wrap=True)
            ax.set_xlabel('')
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            axes_color = self.axes_color
            for spin in ax.spines:
                ax.spines[spin].set_visible(True)
                ax.spines[spin].set_color(axes_color)

    def plot_subplot(self, agent_context: core.AgentContext,
                     xvalues: List[int], yvalues: List[Union[float, Tuple[float, float, float]]], ylabel: str,
                     xlabel: str = None, xlim: Optional[Tuple[float, float]] = None,
                     yscale: str = 'linear', ylim: Optional[Tuple[float, float]] = None,
                     color: str = 'blue'):
        """Draws the graph given by xvalues, yvalues (including x- & y-axes) .

        Attributes:
            agent_context: context containing the figure to plot to
            xvalues: the graphs x-values (must have same length as y-values)
            yvalues: the graphs y-values or (min,y,max)-tuples (must have same length as x-values)
            ylim: (min,max) for the x-axes
            xlabel: label of the x-axes
            ylabel: label of the y-axes
            yscale: scale of the y-axes ('linear', 'log',...)
            ylim: (min,max) for the y-axes (or None)
            color: the graphs color (must be the name of a matplotlib color)
        """
        if not xlim:
            xmin = 0
            xmax = 1
            if agent_context.is_play:
                xmin = 1
                xmax = agent_context.play.episodes_done
            if agent_context.is_train or agent_context.is_eval:
                xmin = 0
                xmax = agent_context.train.episodes_done_in_training
            xlim = (xmin, xmax)
        if xlabel is None:
            xlabel = 'episodes'
            if agent_context.is_play:
                xlabel = 'episodes played'
            if agent_context.is_eval or agent_context.is_train:
                xlabel = 'episodes trained'
        self.clear_plot(agent_context)
        self.plot_axes(xlim=xlim, xlabel=xlabel, ylabel=ylabel, yscale=yscale, ylim=ylim)
        self.plot_values(agent_context=agent_context, xvalues=xvalues, yvalues=yvalues, color=color)

    def plot_values(self, agent_context: core.AgentContext,
                    xvalues: List[int], yvalues: List[Union[float, Tuple[float, float, float]]],
                    color: Optional[str] = 'blue', marker: str = None, pause: bool = True):
        """Draws the graph given by xvalues, yvalues.

        Attributes:
            agent_context: context containing the figure to plot to
            xvalues: the graphs x-values (must have same length as y-values)
            yvalues: the graphs y-values or (min,y,max)-tuples (must have same length as x-values)
            color: the graphs color (must be the name of a matplotlib color) or None
            marker: if None a marker is used if the graph has just a few points.
            pause: pause to redraw the plot.
        """
        assert xvalues is not None
        assert yvalues is not None
        assert len(xvalues) == len(yvalues), "xvalues do not match yvalues"

        pyc: core.PyPlotContext = agent_context.pyplot

        # extract min / max and y values if yvalues is of the form [(min,y,max),...)
        yminvalues = None
        ymaxvalues = None
        if len(yvalues) > 0 and isinstance(yvalues[0], tuple):
            ymaxvalues = [t[2] for t in yvalues]
            yminvalues = [t[0] for t in yvalues]
            yvalues = [t[1] for t in yvalues]

        # plot values
        marker = marker
        if marker is None and len(xvalues) < 10:
            marker = 'o'
        fill_alpha = 0.1
        if pyc.is_jupyter_active:
            if yminvalues is not None:
                self.axes.fill_between(xvalues, yminvalues, ymaxvalues, color=color, alpha=fill_alpha)
            self.axes.plot(xvalues, yvalues, color=color, marker=marker)
        else:
            if yminvalues is not None:
                plt.fill_between(xvalues, yminvalues, ymaxvalues, color=color, alpha=fill_alpha)
            plt.plot(xvalues, yvalues, color=color, marker=marker)
            if pause:
                plt.pause(0.01)


# noinspection DuplicatedCode
class Actions(_PlotCallback):

    def __init__(self, num_steps_between_plot=100):
        """Plots a histogram of the actions taken during play or during the last evaluation period.

        Args:
            num_steps_between_plot: num of steps to play before plot is updated.
        """
        super().__init__(plot_type=core.PlotType.PLAY_STEP | core.PlotType.PLAY_EPISODE | core.PlotType.TRAIN_EVAL)
        assert num_steps_between_plot > 0
        self._actions: List[float] = []
        self._num_steps_between_plot = num_steps_between_plot

    def _reset(self):
        self._actions: List[float] = []

    def on_play_begin(self, agent_context: core.AgentContext):
        super().on_play_begin(agent_context)
        self._reset()

    def on_play_end(self, agent_context: core.AgentContext):
        super().on_play_end(agent_context)
        self._reset()

    def plot(self, agent_context: core.AgentContext, plot_type: core.PlotType):
        try:
            pc = agent_context.play
            is_plot = True
            xlabel = 'actions'
            if plot_type & core.PlotType.TRAIN_EVAL != core.PlotType.NONE:
                xlabel = 'actions taken during last evaluation period'
            if plot_type & core.PlotType.PLAY_STEP != core.PlotType.NONE:
                is_plot = (pc.steps_done_in_episode % self._num_steps_between_plot == 0)
            if is_plot:
                self.clear_plot(agent_context)
                self.plot_axes(xlabel=xlabel, ylabel='count')
                self.axes.hist(pc.actions.values())
        except:
            self.plot_text(f'Failed to create the actions histogram.\n')


class Loss(_PlotCallback):

    def __init__(self, yscale: str = 'symlog', ylim: Optional[Tuple[float, float]] = None):
        """Plots the loss resulting from each iterations policy training.

        Hints:
        o for actro-critic agents the loss from training the actor- and critic-networks are plotted
            along with the total loss.

        Args:
            yscale: scale of the y-axes ('linear', 'symlog',...)
            ylim: (min,max) for the y-axes
        """
        super().__init__(plot_type=core.PlotType.TRAIN_ITERATION)
        self.ylim = ylim
        self.yscale = yscale

    def plot(self, agent_context: core.AgentContext, plot_type: core.PlotType):
        ac = agent_context
        tc = ac.train
        xvalues = list(tc.loss.keys())
        self.clear_plot(agent_context)
        lossvalues = list(tc.loss.values())
        if self._is_nan(lossvalues):
            self.plot_text('plot not available for this backend.')
        else:
            self.plot_axes(xlim=(0, tc.episodes_done_in_training), xlabel='episodes trained',
                           ylim=self.ylim, ylabel='loss', yscale=self.yscale)
            if isinstance(tc, core.ActorCriticTrainContext):
                acc: core.ActorCriticTrainContext = tc
                self.plot_values(agent_context=ac, xvalues=xvalues, yvalues=lossvalues, color='indigo', pause=False)
                self.plot_values(agent_context=ac, xvalues=xvalues, yvalues=list(acc.actor_loss.values()), color='g',
                                 pause=False)
                self.plot_values(agent_context=ac, xvalues=xvalues, yvalues=list(acc.critic_loss.values()), color='b')
                self.axes.legend(('total', 'actor', 'critic'))
            else:
                self.plot_values(agent_context=ac, xvalues=xvalues, yvalues=lossvalues, color='indigo')


class Rewards(_PlotCallback):

    def __init__(self, yscale: str = 'linear', ylim: Optional[Tuple[float, float]] = None):
        """Plots the sum of rewards observed during policy evaluation.

        Args:
            yscale: scale of the y-axes ('linear', 'symlog',...)
            ylim: (min,max) for the y-axes
        """
        super().__init__(core.PlotType.TRAIN_ITERATION | core.PlotType.TRAIN_EVAL | core.PlotType.PLAY_EPISODE)
        self.ylim = ylim
        self.yscale = yscale

    def plot(self, agent_context: core.AgentContext, plot_type: core.PlotType):
        xvalues = yvalues = []
        ylabel = 'sum of rewards'
        if agent_context.is_train or agent_context.is_eval:
            tc = agent_context.train
            xvalues = list(tc.eval_rewards.keys())
            yvalues = list(tc.eval_rewards.values())
            ylabel = 'Ø sum of rewards'
        if agent_context.is_play:
            pc = agent_context.play
            xvalues = list(pc.sum_of_rewards.keys())
            yvalues = list(pc.sum_of_rewards.values())
        if xvalues:
            self.plot_subplot(agent_context, color='green', ylim=self.ylim, yscale=self.yscale,
                              xvalues=xvalues, yvalues=yvalues, ylabel=ylabel)


class State(_PlotCallback):
    """Renders the gym state as a plot to the pyplot figure using gym.render('rgb_array').

        During training only the last state of the last game evaluation is plotted.
        During play all states are plotted.
    """

    def __init__(self, mode='rgb_array'):
        """
        Args:
            mode: the render mode passed to gym.render(), yielding an rgb_array
        """
        super().__init__(plot_type=core.PlotType.PLAY_STEP | core.PlotType.TRAIN_EVAL)
        self._render_mode = mode

    def _plot_rgb_array(self, agent_context: core.AgentContext, rgb_array: np.ndarray):
        """Renders rgb_array to the current subplot."""
        assert rgb_array is not None
        ax = self.axes
        xlabel = ''
        if agent_context.is_eval:
            xlabel = "'done state' of last evaluation episode"
        ax.imshow(rgb_array)
        ax.set_xlabel(xlabel)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        axes_color = self.axes_color
        for spin in ax.spines:
            ax.spines[spin].set_visible(True)
            ax.spines[spin].set_color(axes_color)

    # noinspection PyArgumentList,DuplicatedCode
    def _render_to_rgb_array(self, gym_env: gym.Env, mode: str) -> np.ndarray:
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

    def plot(self, agent_context: core.AgentContext, plot_type: core.PlotType):
        self.clear_plot(agent_context)
        try:
            rgb_array: np.ndarray = self._render_to_rgb_array(agent_context.play.gym_env, self._render_mode)
            self._plot_rgb_array(agent_context, rgb_array)
        except:
            self.plot_text(f'gym.Env.render(mode="{self._render_mode}") failed:\n')


class Steps(_PlotCallback):

    def __init__(self, yscale: str = 'linear', ylim: Optional[Tuple[float, float]] = None):
        """Plots the step counts observed during policy evaluation.

        Args:
            yscale: scale of the y-axes ('linear','log')
            ylim: (min,max) for the y-axes
        """
        super().__init__(core.PlotType.TRAIN_ITERATION | core.PlotType.TRAIN_EVAL | core.PlotType.PLAY_EPISODE)
        self.ylim = ylim
        self.yscale = yscale

    def plot(self, agent_context: core.AgentContext, plot_type: core.PlotType):
        xvalues = yvalues = []
        ylabel = 'steps'
        if agent_context.is_train or agent_context.is_eval:
            tc = agent_context.train
            xvalues = list(tc.eval_steps.keys())
            yvalues = list(tc.eval_steps.values())
            ylabel = 'Ø steps'
        if agent_context.is_play:
            pc = agent_context.play
            xvalues = list(pc.actions.keys())
            yvalues = [len(pc.actions[episode]) for episode in pc.actions.keys()]
        self.plot_subplot(agent_context, color='blue', ylim=self.ylim, yscale=self.yscale,
                          xvalues=xvalues, yvalues=yvalues, ylabel=ylabel)


class StepRewards(_PlotCallback):

    def __init__(self, num_steps_between_plot=100):
        """Plots the sum of rewards up to the current step during play or at the end of an
            evaluation period.

        Args:
            num_steps_between_plot: num of steps to play before plot is updated.
        """
        super().__init__(core.PlotType.PLAY_STEP | core.PlotType.PLAY_EPISODE | core.PlotType.TRAIN_EVAL)
        assert num_steps_between_plot > 0
        self._xy_values: Dict[int, Tuple[List[int], List[float]]] = dict()
        self._xmax: int = 0
        self._num_steps_between_plot = num_steps_between_plot

    def _replot(self, agent_context: core.AgentContext):
        if self._xmax >= 1:
            self.clear_plot(agent_context)
            xlabel = 'steps played'
            if agent_context.is_eval:
                xlabel = 'steps taken during last evaluation period'
            self.plot_axes(xlim=(1, self._xmax), ylabel='sum of rewards', xlabel=xlabel)
            xy_values = list(self._xy_values.values())
            xlast, _ = xy_values[-1]
            for xvalues, yvalues in xy_values:
                pause = (xvalues == xlast)
                self.plot_values(agent_context=agent_context, xvalues=xvalues, yvalues=yvalues,
                                 color=None, marker='', pause=pause)

    def _reset(self):
        self._xy_values = dict()
        self._xmax = 0

    def on_play_begin(self, agent_context: core.AgentContext):
        super().on_play_begin(agent_context)
        self._reset()

    def on_play_end(self, agent_context: core.AgentContext):
        super().on_play_end(agent_context)
        self._reset()

    def plot(self, agent_context: core.AgentContext, plot_type: core.PlotType):
        pc = agent_context.play
        if plot_type & core.PlotType.PLAY_STEP != core.PlotType.NONE:
            episode = pc.episodes_done + 1
            if episode not in self._xy_values:
                self._xy_values[episode] = ([], [])
            xvalues, yvalues = self._xy_values[episode]
            xvalues.append(pc.steps_done_in_episode)
            yvalues.append(pc.sum_of_rewards[episode])
            if pc.steps_done_in_episode > self._xmax:
                self._xmax = pc.steps_done_in_episode
            if (episode == 1 and pc.steps_done == 1) or (pc.steps_done % self._num_steps_between_plot) == 0:
                self._replot(agent_context)
        if plot_type & core.PlotType.PLAY_EPISODE != core.PlotType.NONE:
            self._replot(agent_context)

        if plot_type & core.PlotType.TRAIN_EVAL != core.PlotType.NONE:
            self._xmax = max([len(step_rewards) for step_rewards in pc.rewards.values()])
            for episode in pc.rewards.keys():
                step_rewards = pc.rewards[episode]
                xvalues = list(range(1, len(step_rewards) + 1))
                yvalues = []
                for reward in step_rewards:
                    old_sum = yvalues[-1] if len(yvalues) > 0 else 0
                    yvalues.append(old_sum + reward)
                self._xy_values[episode] = (xvalues, yvalues)
            self._replot(agent_context)


class ToMovie(core._PostProcessCallback):
    """Plots the pyplot figure to an mp4 file

    Attributes:
        fps: frame per seconds
        filepath: the filepath of the mp4 file.
    """

    def __init__(self, fps: Optional[int] = None, filepath: str = None):
        """Writes the ploted graphs and images to the mp4 file given by filepath.

        Args:
            fps: frames per second
            filepath: the filepath of the mp4 file. If None the file is written to a temp file
        """
        super().__init__()
        self.fps = fps
        self._is_filepath_set = filepath is not None
        self.filepath = filepath
        if not self._is_filepath_set:
            self.filepath = self._get_temp_path()
        if not self.filepath.lower().endswith('.mp4'):
            self.filepath = self.filepath + '.mp4'
        self._video = imageio.get_writer(self.filepath, fps=fps) if fps else imageio.get_writer(self.filepath)

    def _close(self, agent_context: core.AgentContext):
        """closes the mp4 file and displays it in jupyter cell (if in a jupyter notebook)"""
        self._video.close()
        self._video = None
        if agent_context.pyplot.is_jupyter_active:
            with open(self.filepath, 'rb') as f:
                video = f.read()
                b64 = base64.b64encode(video)
            if not self._is_filepath_set:
                os.remove(self.filepath)
            result = '''
            <video width="{0}" height="{1}" controls>
                <source src="data:video/mp4;base64,{2}" type="video/mp4">
            Your browser does not support the video tag.
            </video>'''.format(640, 480, b64.decode())
            result = HTML(result)
            # noinspection PyTypeChecker
            clear_output(wait=True)
            # noinspection PyTypeChecker
            display(result)

    def _get_rgb_array(self, agent_context: core.AgentContext) -> np.ndarray:
        """Yields an rgb array representing the current content of the subplots."""
        pyc = agent_context.pyplot
        pyc.figure.canvas.draw()
        result = np.frombuffer(pyc.figure.canvas.tostring_rgb(), dtype='uint8')
        result = result.reshape(pyc.figure.canvas.get_width_height()[::-1] + (3,))
        return result

    def _get_temp_path(self):
        result = os.path.join(tempfile.gettempdir(), tempfile.gettempprefix())
        n = datetime.datetime.now()
        result = result + f'-{n.year % 100:2}{n.month:02}{n.day:02}-{n.hour:02}{n.minute:02}{n.second:02}-' + \
                 f'{n.microsecond:06}'
        return result

    def _write_figure_to_video(self, agent_context: core.AgentContext):
        """Appends the current pyplot figure to the video.

        if an exception occures no frame is added.
        """
        try:
            rgb_array = self._get_rgb_array(agent_context)
            self._video.append_data(rgb_array)
        except:
            pass

    def on_play_episode_end(self, agent_context: core.AgentContext):
        if agent_context._is_plot_ready(core.PlotType.PLAY_EPISODE | core.PlotType.TRAIN_EVAL):
            self._write_figure_to_video(agent_context)

    def on_play_step_end(self, agent_context: core.AgentContext, action, step_result: Tuple):
        if agent_context._is_plot_ready(core.PlotType.PLAY_STEP):
            self._write_figure_to_video(agent_context)

    def on_train_iteration_end(self, agent_context: core.AgentContext):
        if agent_context._is_plot_ready(core.PlotType.TRAIN_ITERATION):
            self._write_figure_to_video(agent_context)

    def on_play_end(self, agent_context: core.AgentContext):
        if agent_context._is_plot_ready(core.PlotType.PLAY_EPISODE):
            self._close(agent_context)

    def on_train_end(self, agent_context: core.AgentContext):
        self._close(agent_context)
