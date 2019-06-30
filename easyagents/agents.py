from logging import INFO, WARNING, getLogger
from easyagents.config import TrainingDuration
from easyagents.config import Logging
from easyagents.easyenv import register
import matplotlib
import matplotlib.pyplot as plt
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
        self.training_average_returns = []
        self.training_losses= []

        self._log = getLogger(name=__name__)
        self._log.setLevel(WARNING)
        if self._logging.log_agent:
            self._log.setLevel(INFO)
        self._log.info( str(self) )

        self._gym_env_name = register(  gym_env_name    = self._gym_env_name, 
                                        log_api         = self._logging.log_gym_api,
                                        log_steps       = self._logging.log_gym_api_steps,
                                        log_reset       = self._logging.log_gym_api_reset   )
        return

    def __str__(self):
        """ yields a human readable representation of the agents/algorithms current configuration
        """
        result = "gym_env_name=" + self._gym_env_name + " fc_layers=" + str(self.fc_layers)
        return result

    def _log_api_call(self, msg):
        self._log.info(msg)
        return

    def compute_avg_return(self ) -> float:
        """ computes the expected sum of rewards for the previously trained policy.

            Note:
            The evaluation is performed on a instance of gym_env_name.
        """
        self._log_api_call(f'executing compute_avg_return(...)')
                    
        sum_rewards = 0.0
        for _ in range(self._training_duration.num_eval_episodes):
            sum_rewards += self.play_episode()
        result = sum_rewards / self._training_duration.num_eval_episodes
        self._log_api_call(f'completed compute_avg_return(...) = {float(result):.3f}')
        return result


    def play_episode (self, callback = None) -> float:
        """ Plays a full episode using the previously trained policy, returning the sum of rewards over the full episode. 

            Args:
            callback    : callback(action,state,reward,done,info) is called after each step.
                          if the callback yields True, the episode is aborted.      
        """
        self._log_api_call(f'executing play_episode(...)')
        self._log_api_call(f'completed play_episode(...)')
        return 0


    def plot_average_returns(self):
        """ produces a matlib.pyplot plot showing the average returns during training.

            Note:
            To see the plot you should call this method from IPython / jupyter notebook.
        """
        episodes_per_value = self._training_duration.num_iterations_between_eval * self._training_duration.num_episodes_per_iteration
        value_count = len(self.training_average_returns)
        steps = range(0, value_count*episodes_per_value, episodes_per_value)
        plt.plot(steps, self.training_average_returns )
        plt.ylabel('average returns')
        plt.xlabel('episodes')
        

    def plot_losses(self):
        """ produces a matlib.pyplot plot showing the losses during training.

            Note:
            To see the plot you should call this method from IPython / jupyter notebook.
        """
        episodes_per_value = self._training_duration.num_episodes_per_iteration
        value_count = len(self.training_losses)
        steps = range(0, value_count*episodes_per_value, episodes_per_value)
        plt.plot(steps, self.training_losses )
        plt.ylabel('losses')
        plt.xlabel('episodes')


