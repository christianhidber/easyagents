import easyagents.core as core


class Fast(core.AgentCallback):
    """Train for max. 100 = 20 x x episodes with max 100 steps per episodes."""

    def __init__(self,
                 num_iterations=20,
                 max_steps_per_episode=100,
                 num_episodes_per_iteration=5,
                 num_iterations_between_eval=3,
                 num_episodes_per_eval=5):
        self.num_iterations = num_iterations
        self.max_steps_per_episode = max_steps_per_episode
        self.num_episodes_per_iteration = num_episodes_per_iteration
        self.num_iterations_between_eval = num_iterations_between_eval
        self.num_episodes_per_eval = num_episodes_per_eval

    def on_play_begin(self, agent_context: core.AgentContext):
        agent_context.play.num_episodes = self.num_episodes_per_iteration
        agent_context.play.max_steps_per_episode = self.max_steps_per_episode

    def on_train_begin(self, agent_context: core.AgentContext):
        tc = agent_context.train
        tc.num_iterations = self.num_iterations
        tc.num_episodes_per_iteration = self.num_episodes_per_iteration
        tc.num_iterations_between_eval = self.num_iterations_between_eval
        tc.num_episodes_per_eval = self.num_episodes_per_eval
        tc.max_steps_per_episode = self.max_steps_per_episode


class _SingleEpisode(Fast):
    """Train / Play only for 1 episode (no evaluation in training, max. 10 steps by default)."""

    def __init__(self, max_steps_per_episode: int = 10):
        super().__init__(num_iterations=1, num_episodes_per_iteration=1, num_iterations_between_eval=0,
                         max_steps_per_episode=max_steps_per_episode)


class _SingleIteration(Fast):
    """Train / play for a single iteration with 3 episodes, and evaluation of 2 episodes"""

    def __init__(self, max_steps_per_episode: int = 25):
        super().__init__(num_iterations=1, num_episodes_per_iteration=3, max_steps_per_episode=max_steps_per_episode,
                         num_episodes_per_eval=2, num_iterations_between_eval=2)
