from typing import Optional, List, Tuple

import easyagents.core as core
import matplotlib.pyplot as plt
import numpy as np
import imageio

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


class _FigurePreProcess(core.AgentCallback):
    """Initializes the matplotlib agent_context.figure"""

    def on_play_begin(self, agent_context: core.AgentContext):
        self.create_figure((agent_context))

    def on_train_begin(self, agent_context: core.AgentContext):
        self.create_figure((agent_context))

    def on_play_episode_end(self, agent_context: core.AgentContext):
        self.clear_all_axes(agent_context)

    def on_train_iteration_end(self, agent_context: core.AgentContext):
        self.clear_all_axes(agent_context)

    def create_figure(self, agent_context: core.AgentContext):
        x_set, y_set = 17, 5
        figure = plt.figure("_EasyAgents", figsize=(x_set, y_set))
        agent_context.figure = figure

    def clear_all_axes(self, agent_context: core.AgentContext):
        def postprocess_main_begin(self, agent_context: core.AgentContext):
            # make sure the plotting takes place on our figure
            figure = agent_context.figure
            if _is_jupyter_active:
                clear_output(wait=True)
                for ax in figure.axes:
                    ax.cla()
            else:
                plt.figure(figure.number)
                plt.cla()


class _FigurePostProcess(core.AgentCallback):
    """Plots the matplotlib agent_context.figure"""

    def __init__(self):
        self._isfirstcall = True

    def on_play_episode_end(self, agent_context: core.AgentContext):
        self.show(agent_context)

    def on_train_iteration_end(self, agent_context: core.AgentContext):
        self.show(agent_context)

    def show(self, agent_context: core.AgentContext):
        # avoid an initial doubled figure output
        if _is_jupyter_active and not self._isfirstcall:
            # noinspection PyTypeChecker
            display(agent_context.figure)
            self._isfirstcall = False
        plt.pause(0.01)


class PlotLossX(core.AgentCallback):

    def __init__(self):
        self._axes: Optional[plt.Axes] = None

    def on_train_begin(self, agent_context: core.AgentContext):
        self._axes = agent_context.figure.add_subplot(1, 1, 1)

    def on_play_episode_end(self, agent_context: core.AgentContext):
        self.plot(agent_context)

    def on_train_iteration_end(self, agent_context: core.AgentContext):
        self.plot(agent_context)

    def set_current_axes(self, agent_context: core.AgentContext):
        # under unittest the current figure seems not to be available anymore
        figure = agent_context.figure
        if not _is_jupyter_active:
            plt.figure(figure.number)
            if figure == plt.gcf():
                plt.sca(self._axes)


class PlotLoss(core.AgentCallback):

    def __init__(self):
        self._is_initialized = False

    def on_train_begin(self, agent_context: core.AgentContext):
        self._is_initialized = False

    def on_train_iteration_end(self, agent_context: core.AgentContext):
        agent_context.figure = self._plot_episodes(agent_context=agent_context,
                                                   is_jupyter_display=self._is_initialized)
        self._is_initialized= True

    def _plot_episodes(self, ylim: List[Tuple[float, float]] = None, scale: List[str] = None,
                       is_jupyter_display: bool = False,
                       rgb_array: np.ndarray = None, agent_context: core.AgentContext = None):
        """ Draws a figure with 3 subplots. If rgb_array is not None then an additional
            subplot with an image of the rgb_array is added.

            if is_jupyter_display_figure is set, then display(figure) is called if we are running
            inside a jupyter notebook. Hereby an initial doubled figure output is avoided.
        """
        assert ylim is None or len(ylim) == 3, "ylim must contain an (float,float) for each of the 3 plots."
        assert scale is None or len(scale) == 3, "scale must contain an 'linear' or 'log' for each of the 3 plots."
        figure = agent_context.figure
        tc = agent_context.train
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
        episodes_per_value = tc.num_episodes_per_iteration
        xlim = episodes_per_value * (len(tc.loss) - 1)
        xlim = 1 if xlim <= 1 else xlim
        self._subplot(axes=axes[0 + offset], yvalues=list(tc.loss.values()), episodes_per_value=episodes_per_value,
                      ylabel='loss', ylim=ylim[0], scale=scale[0], xlim=xlim, color='indigo')

        episodes_per_value = tc.num_episodes_per_iteration * tc.num_iterations_between_eval
        self._subplot(axes=axes[1 + offset], yvalues=list(tc.eval_rewards.values()),
                      episodes_per_value=episodes_per_value,
                      ylabel='sum of rewards', ylim=ylim[1], scale=scale[1], xlim=xlim, color='g')
        self._subplot(axes=axes[2 + offset], yvalues=list(tc.eval_steps.values()), episodes_per_value=episodes_per_value,
                      ylabel='steps', ylim=ylim[2], scale=scale[2], xlim=xlim, color='b')

        # make sure the plots are presented to the user
        if _is_jupyter_active and is_jupyter_display:
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
