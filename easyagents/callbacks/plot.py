from typing import Optional, List, Tuple, Union

import easyagents.core as core
import matplotlib.pyplot as plt
import imageio
import math

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


class _PlotPreProcess(core.AgentCallback):
    """Initializes the matplotlib agent_context.figure"""

    def _clear(self, agent_context: core.AgentContext):
        pyc = agent_context.pyplot

        if pyc.is_jupyter_active:
            clear_output(wait=True)
        else:
            plt.figure(pyc.figure.number)
            plt.cla()

    def on_train_begin(self, agent_context: core.AgentContext):
        pyc = agent_context.pyplot
        pyc.is_jupyter_active = _is_jupyter_active
        pyc._call_jupyter_display = False

        if pyc.figure is None:
            x_set, y_set = 17, 5
            pyc.figure = plt.figure("_EasyAgents", figsize=(x_set, y_set))
        for ax in pyc.figure.axes:
            pyc.figure.delaxes(ax)

    def on_train_end(self, agent_context: core.AgentContext):
        self._clear(agent_context)

    def on_train_iteration_end(self, agent_context: core.AgentContext):
        self._clear(agent_context)


class _PlotPostProcess(core.AgentCallback):
    """Plots the matplotlib agent_context.figure"""

    def _refresh(self, agent_context: core.AgentContext):
        """Fixes the layout of multiple subplots and refreshs the display."""
        pyc = agent_context.pyplot

        count = len(pyc.figure.axes)
        rows = math.ceil( count/pyc.max_columns )
        columns = math.ceil( count/rows )
        for i in range(count):
            pyc.figure.axes[i].change_geometry(rows,columns,i+1)
        pyc.figure.tight_layout()

        if pyc.is_jupyter_active and pyc._call_jupyter_display:
            # noinspection PyTypeChecker
            display(pyc.figure)
        plt.pause(0.01)
        pyc._call_jupyter_display = True

    def on_train_end(self, agent_context: core.AgentContext):
        self._refresh(agent_context)

    def on_train_iteration_end(self, agent_context: core.AgentContext):
        self._refresh(agent_context)


class _PlotCallback(core.AgentCallback):
    """Base class of plyplot callbacks generating a plot after a trained iteration or an episode played.

        Attributes:
            axes: the subplot to plot onto
            is_plot_trained_iteration: True if a plot is created during train
            is_plot_played_episode: True if a plot is created during play
    """

    def __init__(self, is_plot_trained_iteration: bool = False, is_plot_played_episode: bool = False):
        self.axes = None
        self.is_plot_trained_iteration = is_plot_trained_iteration
        self.is_plot_played_episode = is_plot_played_episode
        pass

    def _create_subplot(self, agent_context: core.AgentContext):
        pyc = agent_context.pyplot
        count = len(pyc.figure.axes) + 1
        rows = math.ceil( count/pyc.max_columns )
        columns = math.ceil( count/rows )
        self.axes = pyc.figure.add_subplot(rows,columns,count)

    def _set_current_axes(self, agent_context: core.AgentContext):
        pyc = agent_context.pyplot
        if pyc.is_jupyter_active:
            self.axes.cla()
        else:
            if pyc.figure == plt.gcf():
                plt.sca(self.axes)

    def on_play_begin(self, agent_context: core.AgentContext):
        if self.is_plot_played_episode:
            self._create_subplot(agent_context)

    def on_play_episode_end(self, agent_context: core.AgentContext):
        if self.is_plot_played_episode:
            self._set_current_axes(agent_context)
            self.plot_played_episode(agent_context)

    def on_train_begin(self, agent_context: core.AgentContext):
        if self.is_plot_trained_iteration:
            self._create_subplot(agent_context)

    def on_train_end(self, agent_context: core.AgentContext):
        if self.is_plot_trained_iteration:
            self._set_current_axes(agent_context)
            self.plot_trained_iteration(agent_context)

    def on_train_iteration_end(self, agent_context: core.AgentContext):
        if self.is_plot_trained_iteration:
            self._set_current_axes(agent_context)
            self.plot_trained_iteration(agent_context)

    def plot_axes(self,
                  xvalues: List[int], yvalues: List[Union[float, Tuple[float, float, float]]], ylabel: str,
                  xlabel: str = 'episodes', yscale: str = 'linear', ylim: Optional[Tuple[float, float]] = None):
        """Draws the graph given by xvalues, yvalues.

        Attributes:
            xvalues: the graphs x-values (must have same length as y-values)
            yvalues: the graphs y-values or (min,y,max)-tuples (must have same length as x-values)
            xlabel: label of the x-axes
            ylabel: label of the y-axes
            yscale: scale of the y-axes ('linear' or 'log')
            ylim: (min,max) for the y-axes
        """
        assert xvalues
        assert yvalues
        assert len(xvalues) == len(yvalues), "xvalues do not match yvalues"
        assert ylabel
        assert yscale == 'linear' or yscale == 'log'

        xmax = max(xvalues)
        xmax = 1 if xmax <= 1 else xmax

        # setup subplot (axes, labels, colors)
        axes_color = 'grey'
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.set_xlim(0, xmax)
        self.axes.spines['top'].set_visible(False)
        self.axes.spines['right'].set_visible(False)
        self.axes.spines['bottom'].set_color(axes_color)
        self.axes.spines['left'].set_color(axes_color)
        self.axes.grid(color=axes_color, linestyle='-', linewidth=0.25, alpha=0.5)
        if ylim is not None:
            self.axes.set_ylim(ylim)
        self.axes.set_yscale(yscale)

    def plot_to_subplot(self, agent_context: core.AgentContext,
                        xvalues: List[int], yvalues: List[Union[float, Tuple[float, float, float]]], ylabel: str,
                        xlabel: str = 'episodes', yscale: str = 'linear', ylim: Optional[Tuple[float, float]] = None,
                        color: str = 'blue'):
        """Draws the graph given by xvalues, yvalues.

        Attributes:
            agent_context: context containing the figure to plot to
            xvalues: the graphs x-values (must have same length as y-values)
            yvalues: the graphs y-values or (min,y,max)-tuples (must have same length as x-values)
            xlabel: label of the x-axes
            ylabel: label of the y-axes
            yscale: scale of the y-axes ('linear' or 'log')
            ylim: (min,max) for the y-axes
            color: the graphs color (must be the name of a matplotlib color)
        """
        self.plot_axes(xvalues=xvalues, yvalues=yvalues, ylabel=ylabel,
                       xlabel=xlabel, yscale=yscale, ylim=ylim)
        self.plot_values(agent_context=agent_context, xvalues=xvalues, yvalues=yvalues, color=color)

    def plot_values(self, agent_context: core.AgentContext,
                    xvalues: List[int], yvalues: List[Union[float, Tuple[float, float, float]]],
                    color: str = 'blue'):
        """Draws the graph given by xvalues, yvalues.

        Attributes:
            agent_context: context containing the figure to plot to
            xvalues: the graphs x-values (must have same length as y-values)
            yvalues: the graphs y-values or (min,y,max)-tuples (must have same length as x-values)
            color: the graphs color (must be the name of a matplotlib color)
        """
        assert xvalues
        assert yvalues
        assert len(xvalues) == len(yvalues), "xvalues do not match yvalues"

        pyc = agent_context.pyplot

        # extract min / max and y values if yvalues is of the form [(min,y,max),...)
        yminvalues = None
        ymaxvalues = None
        if len(yvalues) > 0 and isinstance(yvalues[0], tuple):
            ymaxvalues = [t[2] for t in yvalues]
            yminvalues = [t[0] for t in yvalues]
            yvalues = [t[1] for t in yvalues]

        # plot values
        fill_alpha = 0.1
        if pyc.is_jupyter_active:
            if yminvalues is not None:
                self.axes.fill_between(xvalues, yminvalues, ymaxvalues, color=color, alpha=fill_alpha)
            self.axes.plot(xvalues, yvalues, color=color)
        else:
            if pyc.figure == plt.gcf():
                plt.sca(self.axes)
            if yminvalues is not None:
                plt.fill_between(xvalues, yminvalues, ymaxvalues, color=color, alpha=fill_alpha)
            plt.plot(xvalues, yvalues, color=color)
            plt.pause(0.01)

    def plot_played_episode(self, agent_context: core.AgentContext):
        """Plots a graph after each episode during play."""
        pass

    def plot_trained_iteration(self, agent_context: core.AgentContext):
        """Plots a graph after each iteration during train."""
        pass


class PlotLoss(_PlotCallback):

    def __init__(self):
        super().__init__(is_plot_trained_iteration=True)

    def plot_trained_iteration(self, agent_context: core.AgentContext):
        tc = agent_context.train
        self.plot_to_subplot(agent_context, color='indigo',
                             xvalues=list(tc.loss.keys()),
                             yvalues=list(tc.loss.values()), ylabel='loss', yscale='log')


class PlotRewards(_PlotCallback):

    def __init__(self):
        super().__init__(is_plot_trained_iteration=True)

    def plot_trained_iteration(self, agent_context: core.AgentContext):
        tc = agent_context.train
        self.plot_to_subplot(agent_context, color='green',
                             xvalues=list(tc.eval_rewards.keys()),
                             yvalues=list(tc.eval_rewards.values()), ylabel='sum of rewards')


class PlotSteps(_PlotCallback):

    def __init__(self):
        super().__init__(is_plot_trained_iteration=True)

    def plot_trained_iteration(self, agent_context: core.AgentContext):
        tc = agent_context.train
        self.plot_to_subplot(agent_context, color='blue',
                             xvalues=list(tc.eval_steps.keys()),
                             yvalues=list(tc.eval_steps.values()), ylabel='steps')
