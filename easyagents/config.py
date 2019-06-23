#
# collection of classes to hold the easyagent's configurations - grouped by topic
# (like training duration, plotting, etc)
#

class TrainingDuration(object):
    """Immutable class to configure the agents runtime behaviour in terms of
       iterations and episodes.
    """

    def __init__(   self,    
                    num_iterations : int = 10,
                    num_episodes_per_iteration : int = 10,
                    max_steps_per_episode : int = 100000,
                    num_epochs_per_iteration : int = 10,
                    num_iterations_between_eval : int = 10,
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
                    max_steps_per_episode : int = 10,
                    num_epochs_per_iteration : int = 1,
                    num_iterations_between_eval : int = 1,
                    num_eval_episodes : int = 1 ):
        super().__init__( num_iterations=num_iterations,
                        num_episodes_per_iteration=num_episodes_per_iteration,
                        max_steps_per_episode=max_steps_per_episode,
                        num_epochs_per_iteration=num_epochs_per_iteration,
                        num_iterations_between_eval=num_iterations_between_eval,
                        num_eval_episodes=num_eval_episodes)

    
