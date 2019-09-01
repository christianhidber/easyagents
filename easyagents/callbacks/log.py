from typing import Tuple
import logging

from easyagents import core


class CountCallbacks(core.AgentCallback):

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

    def on_api_log(self, agent_context: core.AgentContext, api_target: str, log_msg: str):
        self.api_log_count += 1

    def on_log(self, agent_context: core.AgentContext, log_msg: str):
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

    def on_train_begin(self, agent_context: core.AgentContext):
        """Called once at the entry of an agent.train() call. """
        self.train_begin_count += 1

    def on_train_end(self, agent_context: core.AgentContext):
        """Called once before exiting an agent.train() call"""
        self.train_end_count += 1

    def on_train_iteration_begin(self, agent_context: core.AgentContext):
        """Called once at the start of a new iteration. """
        self.train_iteration_begin_count += 1

    def on_train_iteration_end(self, agent_context: core.AgentContext):
        """Called once after the current iteration is completed"""
        self.train_iteration_end_count += 1


class _LogCallbackBase(core.AgentCallback):
    """Base class for Callback loggers"""

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

    def log(self, msg_id: str, *args):
        msg = self._prefix + f'{msg_id:<25}'
        for arg in args:
            if arg is not None:
                msg += str(arg) + ' '
        self._logger.warning(msg)


class LogCallbacks(_LogCallbackBase):
    """Logs all AgentCallback calls to a Logger"""

    def __init__(self, logger: logging.Logger = None, prefix: str = None):
        """Writes all calls to a callback function to logger with the given prefix.

        Args:
            logger: the logger to log (if None a new logger with level debug is created)
            prefix: a string written in front of each log msg
            """
        super().__init__(logger=logger, prefix=prefix)

    def on_api_log(self, agent_context: core.AgentContext, api_target: str, log_msg: str):
        msg = f'{api_target:<30}'
        if log_msg:
            msg += ' ' + log_msg
        self.log('on_api_log', msg)

    def on_log(self, agent_context: core.AgentContext, log_msg: str):
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

    def on_play_begin(self, agent_context: core.AgentContext):
        self.log('on_play_begin', agent_context)

    def on_play_end(self, agent_context: core.AgentContext):
        self.log('on_play_end', agent_context)

    def on_play_episode_begin(self, agent_context: core.AgentContext):
        self.log('on_play_episode_begin', agent_context)

    def on_play_episode_end(self, agent_context: core.AgentContext):
        self.log('on_play_episode_end', agent_context)

    def on_play_step_begin(self, agent_context: core.AgentContext, action):
        self.log('on_play_step_begin', agent_context)

    def on_play_step_end(self, agent_context: core.AgentContext, action, step_result: Tuple):
        self.log('on_play_step_end', agent_context)

    def on_train_begin(self, agent_context: core.AgentContext):
        self.log('on_train_begin', agent_context)

    def on_train_end(self, agent_context: core.AgentContext):
        self.log('on_train_end', agent_context)

    def on_train_iteration_begin(self, agent_context: core.AgentContext):
        self.log('on_train_iteration_begin', agent_context)

    def on_train_iteration_end(self, agent_context: core.AgentContext):
        self.log('on_train_iteration_end', agent_context)


class LogAgent(_LogCallbackBase):
    """Logs agent activities to a python logger."""

    def on_api_log(self, agent_context: core.AgentContext, api_target: str, log_msg: str):
        self.log(api_target, log_msg)

    def on_log(self, agent_context: core.AgentContext, log_msg: str):
        self.log(log_msg)


class LogAgentContext(_LogCallbackBase):
    """Logs the agent context and its subcontexts after every training iteration / episode played """

    def on_play_episode_end(self, agent_context: core.AgentContext):
        self.log(str(agent_context))

    def on_train_iteration_end(self, agent_context: core.AgentContext):
        self.log(str(agent_context))

class LogIteration(_LogCallbackBase):
    """Logs training iteration summaries to a python logger."""

    def log_iteration(self, agent_context: core.AgentContext):
        tc = agent_context.train
        e = tc.episodes_done_in_training
        msg = f'episodes_done={e:<3} '
        if e in tc.loss:
            msg = msg + f'loss={tc.loss[e]:<7.1f} '
            if isinstance(tc, core.ActorCriticTrainContext):
                msg = msg + f'[actor={tc.actor_loss[e]:<7.1f} '
                msg = msg + f'critic={tc.critic_loss[e]:<7.1f}] '
        if e in tc.eval_rewards:
            r = tc.eval_rewards[e]
            msg = msg + f'rewards=({r[0]:.1f},{r[1]:.1f},{r[2]:.1f}) '
        if e in tc.eval_steps:
            s = tc.eval_steps[e]
            msg = msg + f'steps=({s[0]:.1f},{s[1]:.1f},{s[2]:.1f}) '
        self.log(f'iteration {tc.iterations_done_in_training:<2} of {tc.num_iterations} {msg}')

    def on_train_iteration_begin(self, agent_context: core.AgentContext):
        tc = agent_context.train
        if 0 in tc.eval_rewards and tc.episodes_done_in_training == 0:
            self.log_iteration(agent_context)

    def on_train_iteration_end(self, agent_context: core.AgentContext):
        self.log_iteration(agent_context)


class LogStep(_LogCallbackBase):
    """Logs each environment step to a python logger."""

    def on_gym_step_end(self, agent_context: core.AgentContext, action, step_result: Tuple):
        prefix = ''
        tc: core.TrainContext = agent_context.train
        if tc:
            prefix = f'train iteration={tc.iterations_done_in_training:<2} step={tc.steps_done_in_iteration:<4}'
        pc: core.PlayContext = agent_context.play
        if pc:
            prefix = f'play  episode={pc.episodes_done:<2} step={pc.steps_done_in_episode:<5} ' + \
                     f'sum_of_rewards={pc.sum_of_rewards[pc.episodes_done]:<7.1f}'
        (observation, reward, done, info) = step_result
        msg = ''
        if info:
            msg = f' info={msg}'
        self.log(f'{prefix} reward={reward:<5.1f} done={str(done):5} action={action} observation={observation}{msg}')
