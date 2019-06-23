from logging import INFO, WARNING, getLogger
from easyagents.config import TrainingDuration
from easyagents.config import Logging
from easyagents.logenv import register
import gym

class EasyAgent(object):
    """ Abstract base class for all easy reinforcment learning agents.

        Args:
        gym_env_name            : the name of the registered gym environment to use, eg 'CartPole-v0'
        fc_layers               : int tuple defining the number and size of each fully connected layer
        training_duration       : instance of config.TrainingDuration to configure the #episodes used for training.
        learning_rate           : value in (0,1]. Factor by which the impact on the policy update is reduced
                                  for each training step. The same learning rate is used for the value and the policy network.
        reward_discount_gamma   : value in (0,1]. Factor by which a future reward is discounted for each step.
        logging                 : instance of config.Logging to configure the logging behaviour 
    """

    def __init__(   self,
                    gym_env_name: str,
                    training_duration : TrainingDuration = None,
                    fc_layers = None,                    
                    learning_rate : float = 0.001,
                    reward_discount_gamma : float = 1,
                    logging : Logging = None
                     ): 
        if fc_layers is None:
            fc_layers = (75,75)
        if training_duration is None:
            training_duration = TrainingDuration()
        if logging is None:
            logging = Logging()

        assert isinstance(gym_env_name,str), "passed gym_env_name not a string."
        assert gym_env_name != "", "gym environment name is empty."
        assert fc_layers != None, "fc_layers not set"
        assert isinstance(training_duration,TrainingDuration), "training_duration not an instance of easyagents.config.TrainingDuration"
        assert learning_rate > 0, "learning_rate must be in (0,1]"
        assert learning_rate <= 1, "learning_rate must be in (0,1]"
        assert reward_discount_gamma > 0, "reward_discount_gamma must be in (0,1]"
        assert reward_discount_gamma <= 1, "reward_discount_gamma must be in (0,1]"
        assert isinstance(logging,Logging), "logging not an instance of easyagents.config.Logging"

        self._gym_env_name = gym_env_name
        self._training_duration = training_duration
        self.fc_layers = fc_layers
        self._learning_rate = learning_rate
        self._reward_discount_gamma = reward_discount_gamma
        self._logging=logging

        self._log = getLogger(name=__name__)
        self._log.setLevel(WARNING)
        if self._logging.log_agent:
            self._log.setLevel(INFO)
        self._log.info( str(self) )

        if self._logging.log_gym_env:
            self._gym_env_name = register(  gym_env_name    = self._gym_env_name, 
                                            log_steps       = self._logging.log_gym_env_steps,
                                            log_reset       = self._logging.log_gym_env_reset   )
        return

    def _logCall(self, msg):
        self._log.info(msg)
        return

    def __str__(self):
        """ yields a human readable representation of the agents/algorithms current configuration
        """
        result = "gym_env_name=" + self._gym_env_name + " fc_layers=" + str(self.fc_layers)
        return result


