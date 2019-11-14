import math
import sys

import easyagents.core as core
import easyagents.backends.core as bcore


class Best(core.AgentCallback):
    """After each iteration the current policy is saved if average reward is larger than all previous average
        rewards. The policies can be loaded using agents.load()

        Attributes:
            directory: the absolute path of the directory containing the persisted policies.
    """

    def __init__(self, directory: str = None):
        """Saves the best policies (along with the agent definition) in directory.
        If directory is None the policies are written in a temp directory.

        Args:
            directory: the directory to save to, if None a temp directory is created.
        """
        self.directory = directory if directory else bcore._get_temp_path()
        self.directory = bcore._mkdir(directory)
        self._best_reward = sys.float_info.min

    def on_train_iteration_end(self, agent_context: core.AgentContext):
        """saves the current"""

