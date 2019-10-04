import math
import easyagents.core as core


class Fast(core.AgentCallback):
    """Train for small number of episodes / steps in order to do a dry run of the algorithms or callbacks."""

    def __init__(self, num_iterations=None):
        self._num_iterations = num_iterations
        self._num_episodes_per_iteration = 3
        self._max_steps_per_episode = 50

    def on_play_begin(self, agent_context: core.AgentContext):
        agent_context.play.max_steps_per_episode = self._max_steps_per_episode
        if isinstance(agent_context, core.EpisodesTrainContext):
            agent_context.num_episodes_per_iteration = self._num_episodes_per_iteration

    def on_train_begin(self, agent_context: core.AgentContext):
        tc = agent_context.train
        if self._num_iterations is None:
            self._num_iterations = 10
        if isinstance(tc, core.EpisodesTrainContext):
            ec: core.EpisodesTrainContext = tc
            ec.num_episodes_per_iteration = self._num_episodes_per_iteration
            ec.num_epochs_per_iteration = 1
        if isinstance(tc, core.DqnTrainContext):
            dc: core.DqnTrainContext = tc
            if self._num_iterations is None:
                self._num_iterations = 5 * self._num_episodes_per_iteration * self._max_steps_per_episode
            dc.num_steps_buffer_preload = math.ceil(self._num_iterations / 10)
        tc.num_iterations = self._num_iterations
        tc.num_iterations_between_eval = math.ceil(tc.num_iterations/3)
        tc.num_iterations_between_plot = math.ceil(tc.num_iterations_between_eval/4)
        tc.num_episodes_per_eval = self._num_episodes_per_iteration
        tc.max_steps_per_episode = self._max_steps_per_episode


class _SingleEpisode(Fast):
    """Train / Play only for 1 episode (no evaluation in training, max. 10 steps)."""

    def __init__(self):
        super().__init__(num_iterations=1)
        self._num_episodes_per_iteration = 1
        self._max_steps_per_episode = 10

    def on_train_begin(self, agent_context: core.AgentContext):
        super().on_train_begin(agent_context)
        tc = agent_context.train
        if isinstance(tc, core.DqnTrainContext):
            tc.num_iterations = self._max_steps_per_episode
        tc.num_iterations_between_eval = 0
        tc.num_episodes_per_eval = 0


class _SingleIteration(Fast):
    """Train / play for a single iteration with 3 episodes, and evaluation of 2 episodes"""

    def __init__(self):
        super().__init__(num_iterations=1)
