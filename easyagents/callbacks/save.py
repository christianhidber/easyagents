import os
import json
import sys

import easyagents.core as core
import easyagents.backends.core as bcore


class _SaveCallback(core.AgentCallback):
    """Base class for all agent savent callbacks.

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

    def _save_train_context(self,train_context:core.TrainContext, directory : str ):
        """Saves train_context as a jason in the current directory."""
        tc_json_path = os.path.join( directory, 'train_context.json' )
        with open(tc_json_path, 'w') as jsonfile:
            json.dump(train_context, jsonfile, sort_keys=True, indent=2)


class Best(core.AgentCallback):
    """After each policy evaluation the policy is saved if average reward is larger than all previous average
        rewards. The policies can then be loaded using agents.load()

        Attributes:
            directory: the absolute path of the directory containing the persisted policies.
    """

    def __init__(self, directory: str = None):
        """Saves the best policies (along with the agent definition) in directory.
        If directory is None the policies are written in a temp directory.

        Args:
            directory: the directory to save to, if None a temp directory is created.
        """
        super().__init__(directory=directory)
        self._best_avg_reward = sys.float_info.min

    def on_play_end(self, agent_context: core.AgentContext):
        """After evaluation this agent is saved if its average reward is higher than all previous ones
        in this training."""
        if agent_context.is_eval:
            tc = agent_context.train
            min_rewards, avg_reward, max_rewards = tc.eval_rewards[tc.episodes_done_in_training]
            if avg_reward > self._best_avg_reward:
                current_dir = f'after_episode_{tc.episodes_done_in_training}-avg_reward_{avg_reward}'
                current_dir = os.path.join(self.directory, current_dir)
                agent_context._agent_saver(directory=current_dir)
                self._save_train_context(train_context=tc,directory=current_dir)
                self._best_avg_reward = avg_reward
