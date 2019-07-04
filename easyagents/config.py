#
# collection of classes to hold the easyagent's configurations - grouped by topic
# (like training duration, plotting, etc)
#


class Logging(object):
    """Immutable class to define the logging behaviour of the easyagents environment.

        Args:
        log_agent           : true if the agents api calls are logged
        log_gym_api         : true if the gym_env api calls are logged
        log_gym_api_steps   : true if the gym_env.step(...) calls are logged (given log_gym_api is set)
        log_gym_api_reset   : true if the gym_env.reset(...) calls are logged (given log_gym_api is set)
    """
    def __init__(self,
                 log_minimal: bool = True,
                 log_agent : bool = False,
                 log_gym_api : bool = False,
                 log_gym_api_steps : bool = False,
                 log_gym_api_reset : bool = False):
        self.log_minimal = log_minimal
        self.log_agent = log_agent
        self.log_gym_api = log_gym_api
        self.log_gym_api_steps = log_gym_api_steps
        self.log_gym_api_reset = log_gym_api_reset
        return


class LoggingVerbose(Logging):
    """Logging configuration for full logging (all agent api calls and all gym env calls are logged)
    """
    def __init__(   self,
                    log_minimal: bool = True,
                    log_agent : bool = True,
                    log_gym_api : bool = True,
                    log_gym_api_steps : bool = True,
                    log_gym_api_reset : bool = True):
        super().__init__(
                    log_minimal = log_minimal,
                    log_agent=log_agent,
                    log_gym_api=log_gym_api,
                    log_gym_api_steps=log_gym_api_steps,
                    log_gym_api_reset=log_gym_api_reset
        )


class LoggingNormal(Logging):
    """Logging configuration for full logging (all agent api calls and all gym env calls are logged)
    """
    def __init__(   self,
                    log_minimal: bool = True,
                    log_agent : bool = True,
                    log_gym_api : bool = False,
                    log_gym_api_steps : bool = False,
                    log_gym_api_reset : bool = False):
        super().__init__(
                    log_minimal=log_minimal,
                    log_agent=log_agent,
                    log_gym_api=log_gym_api,
                    log_gym_api_steps=log_gym_api_steps,
                    log_gym_api_reset=log_gym_api_reset
        )


class LoggingSilent(Logging):
    """Logging configuration which suppresses all logging
    """
    def __init__(   self,
                    log_minimal: bool = False,
                    log_agent : bool = False,
                    log_gym_api : bool = False,
                    log_gym_api_steps : bool = False,
                    log_gym_api_reset : bool = False):
        super().__init__(
                    log_minimal=log_minimal,
                    log_agent=log_agent,
                    log_gym_api=log_gym_api,
                    log_gym_api_steps=log_gym_api_steps,
                    log_gym_api_reset=log_gym_api_reset
        )


class TrainingDuration(object):
    """Immutable class to configure the agents runtime behaviour in terms of
       iterations and episodes.
    """

    def __init__(   self,    
                    num_iterations : int = 25,
                    num_episodes_per_iteration : int = 10,
                    max_steps_per_episode : int = 500,
                    num_epochs_per_iteration : int = 5,
                    num_iterations_between_eval : int = 5,
                    num_eval_episodes : int = 10 ):
        """ Groups all properties related to the definition of the algorithms runtime

            Args:
                num_iterations                 : number of times the training is started (with fresh data)
                num_episodes_per_iteration     : number of episodes played per training iteration 
                max_steps_per_episode          : maximum number of steps per episode.
                num_epochs_per_iteration       : number of times the data collected for the current iteration  
                                                 (and stored in the replay buffer) is used to retrain the current policy
                num_iterations_between_eval    : number of training iterations before the current policy is evaluated
                num_eval_episodes              : number of episodes played to estimate the average return
        """
        assert num_iterations >= 1, "num_iterations must be >= 1"
        assert num_episodes_per_iteration >= 1, "num_episodes_per_iteration must be >= 1"
        assert max_steps_per_episode >= 1, "max_steps_per_episode must be >= 1"
        assert num_epochs_per_iteration >= 1, "num_epochs_per_iteration must be >= 1"
        assert num_iterations_between_eval >= 1, "num_iterations_between_eval must be >= 1"        
        assert num_eval_episodes >= 1, "num_eval_episodes must be >= 1"  

        self._num_iterations = num_iterations
        self._num_episodes_per_iteration = num_episodes_per_iteration
        self._max_steps_per_episode = max_steps_per_episode
        self._num_epochs_per_iteration = num_epochs_per_iteration
        self._num_iterations_between_eval = num_iterations_between_eval
        self._num_eval_episodes = num_eval_episodes

    def __str__(self):
        """ yields a human readable representation of the agents/algorithms current configuration
        """
        episodes = self._num_iterations * self._num_episodes_per_iteration
        result = f'{episodes}={self._num_iterations}*{self._num_episodes_per_iteration} episodes [max_steps_per_episode={self._max_steps_per_episode}, num_epochs_per_iteration={self._num_epochs_per_iteration}, num_iterations_between_eval={self._num_iterations_between_eval}]'
        return result

    @property
    def num_episodes(self):
        """ yields the total number of episodes played during training.
        """
        return self._num_iterations * self._num_episodes_per_iteration

    @property
    def num_iterations(self):
        return self._num_iterations

    @property
    def max_steps_per_episode(self):
        return self._max_steps_per_episode

    @property
    def num_episodes_per_iteration(self):
        return self._num_episodes_per_iteration

    @property
    def num_epochs_per_iteration(self):
        return self._num_epochs_per_iteration

    @property
    def num_iterations_between_eval(self):
        return self._num_iterations_between_eval

    @property
    def num_eval_episodes(self):
        return self._num_eval_episodes


class TrainingDurationFast(TrainingDuration):
    """TrainingDuration with constructor defaults set to very small values for fast and easy debugging.
    """
    def __init__(   self,    
                    num_iterations : int = 3,
                    num_episodes_per_iteration : int = 3,
                    max_steps_per_episode : int = 500,
                    num_epochs_per_iteration : int = 1,
                    num_iterations_between_eval : int = 1,
                    num_eval_episodes : int = 1 ):
        super().__init__(   num_iterations=num_iterations,
                            num_episodes_per_iteration=num_episodes_per_iteration,
                            max_steps_per_episode=max_steps_per_episode,
                            num_epochs_per_iteration=num_epochs_per_iteration,
                            num_iterations_between_eval=num_iterations_between_eval,
                            num_eval_episodes=num_eval_episodes)


class TrainingDurationSingleEpisode(TrainingDuration):
    """TrainingDuration with constructor defaults set to a single training set.
    """
    def __init__(   self,    
                    num_iterations : int = 1,
                    num_episodes_per_iteration : int = 1,
                    max_steps_per_episode : int = 500,
                    num_epochs_per_iteration : int = 1,
                    num_iterations_between_eval : int = 1,
                    num_eval_episodes : int = 1 ):
        super().__init__(   num_iterations=num_iterations,
                            num_episodes_per_iteration=num_episodes_per_iteration,
                            max_steps_per_episode=max_steps_per_episode,
                            num_epochs_per_iteration=num_epochs_per_iteration,
                            num_iterations_between_eval=num_iterations_between_eval,
                            num_eval_episodes=num_eval_episodes)
