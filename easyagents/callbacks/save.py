import os
import sys
from typing import List, Tuple

import easyagents.core as core
import easyagents.backends.core as bcore


class _SaveCallback(core.AgentCallback):
    """Base class for all agent savent callbacks.

        Attributes:
            directory: the absolute path of the directory containing the persisted policies.
            saved_agents: list of tuples (episode, avg_rewards, directory) for each saved agent
    """

    def __init__(self, directory: str = None):
        """Saves the best policies (along with the agent definition) in directory.
        If directory is None the policies are written in a temp directory.

        Args:
            directory: the directory to save to, if None a temp directory is created.
        """
        directory = directory if directory else bcore._get_temp_path()
        self.directory: str = bcore._mkdir(directory)
        self.saved_agents: List[Tuple[int, float, str]] = []

    def __str__(self):
        return self.directory

    def _save(self, agent_context: core.AgentContext):
        """Saves the current policy in directory."""
        assert agent_context
        assert agent_context.train, "TrainContext not set."

        tc = agent_context.train
        min_rewards, avg_reward, max_rewards = tc.eval_rewards[tc.episodes_done_in_training]
        current_dir = f'episode_{tc.episodes_done_in_training}-avg_reward_{avg_reward}'
        current_dir = os.path.join(self.directory, current_dir)
        agent_context._agent_saver(directory=current_dir)
        self.saved_agents.append((tc.episodes_done_in_training, avg_reward, current_dir))


class Best(_SaveCallback):
    """After each policy evaluation the policy is saved if average reward is larger than all previous average
        rewards. The policies can then be loaded using agents.load()

        Attributes:
            directory: the absolute path of the directory containing the persisted policies.
            saved_agents: list of tuples (episode, avg_rewards, directory) for each saved agent
    """

    def __init__(self, directory: str = None):
        """Saves the best policies (along with the agent definition) in directory.
        If directory is None the policies are written in a temp directory.

        Args:
            directory: the directory to save to, if None a temp directory is created.
        """
        super().__init__(directory=directory)
        self._best_avg_reward = -sys.float_info.max

    def on_play_end(self, agent_context: core.AgentContext):
        if agent_context.is_eval:
            tc = agent_context.train
            min_rewards, avg_reward, max_rewards = tc.eval_rewards[tc.episodes_done_in_training]
            if avg_reward > self._best_avg_reward:
                self._save(agent_context=agent_context)
                self._best_avg_reward = avg_reward


class Every(_SaveCallback):
    """Saves the current policy every n evaluations.

        Attributes:
            directory: the absolute path of the directory containing the persisted policies.
            saved_agents: list of tuples (episode, avg_rewards, directory) for each saved agent
    """

    def __init__(self, num_evals_between_save: int = 1, directory: str = None):
        """Saves the current policy every n evaluations. In terms of episodes: the policy is saved every

        num_eval_between_saves * num_iterations_between_eval * num_episodes_per_iteration

        Args:
            num_evals_between_save: the number of evaluations between saves.
        """
        assert num_evals_between_save > 0
        super().__init__(directory=directory)
        self.num_evals_between_save: int = num_evals_between_save

    def on_play_end(self, agent_context: core.AgentContext):
        if agent_context.is_eval:
            tc = agent_context.train
            if tc.num_iterations % tc.num_iterations_between_eval == 0:
                evals = tc.num_iterations / tc.num_iterations_between_eval
                if evals % self.num_evals_between_save == 0:
                    self._save(agent_context=agent_context)
