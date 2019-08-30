import easyagents.core as core


class Fast(core.AgentCallback):
    """Train for max. 100 = 10 x 10 episodes with max 100 steps per episodes."""

    def on_play_begin(self, agent_context: core.AgentContext):
        agent_context.play.num_episodes = 10
        agent_context.play.max_steps_per_episode = 100

    def on_train_begin(self, agent_context: core.AgentContext):
        train = agent_context.train
        train.num_iterations = 10
        train.num_episodes_per_iteration = 10
        train.num_iterations_between_eval = 3
        train.num_episodes_per_eval = 3
        train.max_steps_per_episode = 100


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
