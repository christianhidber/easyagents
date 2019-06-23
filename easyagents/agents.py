import logging
from easyagents.config import TrainingDuration

class EasyAgent(object):
    """ Abstract base class for all easy reinforcment learning agents.

        Args:
        gym_env_name            : the name of the registered gym environment to use, eg 'CartPole-v0'
        fc_layers               : int tuple defining the number and size of each fully connected layer
        training_duration       : instance of TrainingDuration to configure the #episodes used for training.
        learning_rate           : value in (0,1]. Factor by which the impact on the policy update is reduced
                                  for each training step. The same learning rate is used for the value and the policy network.
        reward_discount_gamma   : value in (0,1]. Factor by which a future reward is discounted for each step.
    """

    def __init__(   self,
                    gym_env_name: str,
                    training_duration : TrainingDuration = None,
                    fc_layers = None,                    
                    learning_rate : float = 0.001,
                    reward_discount_gamma : float = 1 ): 
        if fc_layers is None:
            fc_layers = (75,75)
        if training_duration is None:
            training_duration = TrainingDuration()
        assert isinstance(gym_env_name,str), "passed gym_env_name not a string."
        assert gym_env_name != "", "gym environment name is empty."
        assert fc_layers != None, "fc_layers not set"
        assert isinstance(training_duration,TrainingDuration), ""
        assert learning_rate > 0, "learning_rate must be in (0,1]"
        assert learning_rate <= 1, "learning_rate must be in (0,1]"
        assert reward_discount_gamma > 0, "reward_discount_gamma must be in (0,1]"
        assert reward_discount_gamma <= 1, "reward_discount_gamma must be in (0,1]"

        self._gym_env_name = gym_env_name
        self._training_duration = training_duration
        self.fc_layers = fc_layers
        self._learning_rate = learning_rate
        self._reward_discount_gamma = reward_discount_gamma

        self._log = logging.getLogger(name="EazyAgent")
        self._log.setLevel(logging.DEBUG)
        self._log.info( str(self) )
        return


    @property
    def gym_env_name(self) -> str:
        return self._gym_env_name

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @property
    def reward_discount_gamma(self) -> float:
        return self._reward_discount_gamma
        
    @property
    def training_duration(self) -> TrainingDuration:
        return self._training_duration

    def __str__(self):
        """ yields a human readable representation of the agents/algorithms current configuration
        """
        result = "gym_env_name=" + self.gym_env_name + " fc_layers=" + str(self.fc_layers)
        return result


