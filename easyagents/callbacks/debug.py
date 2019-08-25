from typing import Tuple
import logging

from easyagents import core


class Count(core.AgentCallback):

    def __init__(self):
        self.gym_init_begin_count = 0
        self.gym_init_end_count = 0
        self.gym_reset_begin_count = 0
        self.gym_reset_end_count = 0
        self.gym_step_begin_count = 0
        self.gym_step_end_count = 0
        self.api_log_count = 0
        self.log_count = 0
        self.train_begin_count = 0
        self.train_end_count = 0
        self.train_iteration_begin_count = 0
        self.train_iteration_end_count = 0

    def __str__(self):
        return f'gym_init={self.gym_init_begin_count}:{self.gym_init_end_count} ' + \
               f'gym_reset={self.gym_reset_begin_count}:{self.gym_reset_end_count} ' + \
               f'gym_step={self.gym_step_begin_count}:{self.gym_step_end_count}' + \
               f'train={self.train_begin_count}:{self.train_end_count} ' + \
               f'train_iteration={self.train_iteration_begin_count}:{self.train_iteration_end_count}' + \
               f'api_log={self.api_log_count} log={self.log_count} '

    def on_api_log(self, api_context: core.AgentContext, api_target: str, log_msg: str):
        self.api_log_count += 1

    def on_log(self, api_context: core.AgentContext, log_msg: str):
        self.log_count += 1

    def on_gym_init_begin(self, agent_context: core.AgentContext):
        self.gym_init_begin_count += 1

    def on_gym_init_end(self, agent_context: core.AgentContext):
        self.gym_init_end_count += 1

    def on_gym_reset_begin(self, agent_context: core.AgentContext, **kwargs):
        self.gym_reset_begin_count += 1

    def on_gym_reset_end(self, agent_context: core.AgentContext, reset_result: Tuple, **kwargs):
        self.gym_reset_end_count += 1

    def on_gym_step_begin(self, agent_context: core.AgentContext, action):
        self.gym_step_begin_count += 1

    def on_gym_step_end(self, agent_context: core.AgentContext, action, step_result: Tuple):
        self.gym_step_end_count += 1

    def on_train_begin(self, train_context: core.TrainContext):
        """Called once at the entry of an agent.train() call. """
        self.train_begin_count += 1

    def on_train_end(self, train_context: core.TrainContext):
        """Called once before exiting an agent.train() call"""
        self.train_end_count += 1

    def on_train_iteration_begin(self, train_context: core.TrainContext):
        """Called once at the start of a new iteration. """
        self.train_iteration_begin_count += 1

    def on_train_iteration_end(self, train_context: core.TrainContext):
        """Called once after the current iteration is completed"""
        self.train_iteration_end_count += 1


class Log(core.AgentCallback):
    """Logs all AgentCallback calls to a Logger"""

    def __init__(self, logger: logging.Logger = None, prefix: str = None):
        """Writes all calls to a callback function to logger with the given prefix.

        Args:
            logger: the logger to log (if None a new logger with level debug is created)
            prefix: a string written in front of each log msg
            """
        self._logger = logger
        if self._logger is None:
            self._logger = logging.getLogger()
        self._prefix = prefix
        if self._prefix is None:
            self._prefix = ''

    def log(self, id: str, *args):
        msg = self._prefix + f'{id:<25}'
        for arg in args:
            if arg is not None:
                msg += str(arg) + ' '
        self._logger.warning(msg)

    def on_api_log(self, api_context: core.AgentContext, api_target: str, log_msg: str):
        msg = f'{api_target:<30}'
        if log_msg:
            msg += ' ' + log_msg
        self.log('on_api_log', msg)

    def on_log(self, api_context: core.AgentContext, log_msg: str):
        self.log('on_log', log_msg)

    def on_gym_init_begin(self, agent_context: core.AgentContext):
        self.log('on_gym_init_begin', agent_context)

    def on_gym_init_end(self, agent_context: core.AgentContext):
        self.log('on_gym_init_end', agent_context)

    def on_gym_reset_begin(self, agent_context: core.AgentContext, **kwargs):
        self.log('on_gym_reset_begin', agent_context)

    def on_gym_reset_end(self, agent_context: core.AgentContext, reset_result: Tuple, **kwargs):
        self.log('on_gym_reset_end', agent_context)

    def on_gym_step_begin(self, agent_context: core.AgentContext, action):
        self.log('on_gym_step_begin', agent_context)

    def on_gym_step_end(self, agent_context: core.AgentContext, action, step_result: Tuple):
        self.log('on_gym_step_end', agent_context)

    def on_train_begin(self, train_context: core.TrainContext):
        self.log('on_train_begin', train_context)

    def on_train_end(self, train_context: core.TrainContext):
        self.log('on_train_end', train_context)

    def on_train_iteration_begin(self, train_context: core.TrainContext):
        self.log('on_train_iteration_begin', train_context)

    def on_train_iteration_end(self, train_context: core.TrainContext):
        self.log('on_train_iteration_end', train_context)
