import easyagents.core as core


class Fast(core.AgentCallback):
    """Train for max. 100 = 10 x 10 episodes with max 100 steps per episodes."""

    def __init__(self,
                 num_iterations=10,
                 max_steps_per_episode=100,
                 num_episodes_per_iteration=10,
                 num_iterations_between_eval=3,
                 num_episodes_per_eval=3):
        self.num_iterations = num_iterations
        self.max_steps_per_episode = max_steps_per_episode
        self.num_episodes_per_iteration = num_episodes_per_iteration
        self.num_iterations_between_eval = num_iterations_between_eval
        self.num_episodes_per_eval = num_episodes_per_eval

        def on_play_begin(self, agent_context: core.AgentContext):
            agent_context.play.num_episodes = self.num_episodes_per_iteration
            agent_context.play.max_steps_per_episode = self.max_steps_per_episode

    def on_train_begin(self, agent_context: core.AgentContext):
        train = agent_context.train
        train.num_iterations = self.num_iterations
        train.num_episodes_per_iteration = self.num_episodes_per_iteration
        train.num_iterations_between_eval = self.num_iterations_between_eval
        train.num_episodes_per_eval = self.num_episodes_per_eval
        train.max_steps_per_episode = self.max_steps_per_episode


class SingleEpisode(core.AgentCallback):
    """Train / Play only for 1 episode (no evaluation in training, max. 10 steps by default)."""

    def __init__(self, max_steps: int = 10):
        super().__init__()
        self.max_steps = max_steps

    def on_play_begin(self, agent_context: core.AgentContext):
        agent_context.play.num_episodes = 1
        agent_context.play.max_steps_per_episode = self.max_steps

    def on_train_begin(self, agent_context: core.AgentContext):
        train = agent_context.train
        train.num_iterations = 1
        train.num_episodes_per_iteration = 1
        train.num_iterations_between_eval = 0
        train.max_steps_per_episode = self.max_steps
        train.num_iterations_between_eval = 0
