import base64
import os
import tempfile
from logging import INFO, getLogger

import gym
import imageio
import matplotlib.pyplot as plt
import numpy as np

from easyagents.config import Logging
from easyagents.config import TrainingDuration
from easyagents.easyenv import EasyEnv
from easyagents.easyenv import register


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
                    logging : Logging = None    ):
        if fc_layers is None:
            fc_layers = (75, 75)
        if training_duration is None:
            training_duration = TrainingDuration()
        if logging is None:
            logging = Logging()

        assert isinstance(gym_env_name, str), "passed gym_env_name not a string."
        assert gym_env_name != "", "gym environment name is empty."
        assert fc_layers is not None, "fc_layers not set"
        assert isinstance(training_duration, TrainingDuration), "training_duration not an instance of easyagents.config.TrainingDuration"
        assert learning_rate > 0, "learning_rate must be in (0,1]"
        assert learning_rate <= 1, "learning_rate must be in (0,1]"
        assert reward_discount_gamma > 0, "reward_discount_gamma must be in (0,1]"
        assert reward_discount_gamma <= 1, "reward_discount_gamma must be in (0,1]"
        assert isinstance(logging, Logging), "logging not an instance of easyagents.config.Logging"

        self._gym_env_name = gym_env_name
        self._training_duration = training_duration
        self.fc_layers = fc_layers
        self._learning_rate = learning_rate
        self._reward_discount_gamma = reward_discount_gamma
        self._logging = logging
        self.training_average_rewards = []
        self.training_average_steps = []
        self.training_losses = []

        self._log = getLogger(name=__name__)
        self._log.setLevel(INFO)
        self._log_minimal(f'{self}')
        self._log_minimal(f'TrainingDuration {self._training_duration}')

        self._gym_env_name = register(  gym_env_name = self._gym_env_name,
                                        log_api = self._logging.log_gym_api,
                                        log_steps = self._logging.log_gym_api_steps,
                                        log_reset = self._logging.log_gym_api_reset   )
        return

    def __str__(self):
        """ yields a human readable representation of the agents/algorithms current configuration
        """
        result = f'{type(self).__name__} on {self._gym_env_name} [fc_layers={self.fc_layers}, learning_rate={self._learning_rate}'
        if self._reward_discount_gamma < 1:
            result += f', reward_discount_gamma={self._reward_discount_gamma}'
        result += ']'
        return result

    def _log_agent(self, msg):
        if self._logging.log_agent:
            self._log.info(msg)
        return

    def _log_minimal(self, msg):
        if self._logging.log_minimal or self._logging.log_agent:
            self._log.info(msg)
        return

    def _clear_average_rewards_and_steps_log(self):
        """ resets the training logs for the vâvg return, avg steps.
        """
        self.training_average_rewards=[]
        self.training_average_steps=[]

    def _record_average_rewards_and_steps(self):
        """ computes the expected sum of rewards and the expected step count for the previously trained policy.
            and adds them to the training logs
           Note:
           The evaluation is performed on a instance of gym_env_name.
        """
        self._log_agent(f'estimating average rewards and episode lengths for current policy...')
        sum_rewards = 0.0
        sum_steps = 0
        for _ in range(self._training_duration.num_eval_episodes):
            (reward, steps) = self.play_episode()
            sum_rewards += reward
            sum_steps += steps
        avg_rewards: float = sum_rewards / self._training_duration.num_eval_episodes
        avg_steps: float = sum_steps / self._training_duration.num_eval_episodes
        self._log_minimal(f'estimated  avg_reward={float(avg_rewards):.3f}, avg_steps={float(avg_steps):.3f}')
        self.training_average_rewards.append(avg_rewards)
        self.training_average_steps.append(avg_steps)

    def play_episode(self, callback = None) -> (float, int):
        """ Plays a full episode using the previously trained policy, returning the sum of rewards over the full episode. 
            Initially the eval_env.reset is called (callback action set to None)

            Args:
            callback    : callback(gym_env,action,state,reward,done,info) is called after each step.
        """
        return (0.0, 0)

    def plot_average_rewards(self, ylim=None):
        """ produces a matlib.pyplot plot showing the average sum of rewards per episode during training.

            Args:
            ylim    : [ymin,ymax] values for the plot

            Note:
            To see the plot you may call this method from IPython / jupyter notebook.
        """
        episodes_per_value = self._training_duration.num_episodes_per_iteration*self._training_duration.num_iterations_between_eval
        self._plot_episodes(yvalues=self.training_average_rewards, episodes_per_value=episodes_per_value, ylabel='rewards', ylim=ylim )

    def plot_average_steps(self, ylim=None):
        """ produces a matlib.pyplot plot showing the average number of steps per episode during training.

            Args:
            ylim    : [ymin,ymax] values for the plot

            Note:
            To see the plot you may call this method from IPython / jupyter notebook.
        """
        episodes_per_value = self._training_duration.num_episodes_per_iteration * self._training_duration.num_iterations_between_eval
        self._plot_episodes(yvalues=self.training_average_steps, episodes_per_value=episodes_per_value, ylabel='steps', ylim=ylim)
        
    def plot_losses(self, ylim=None ):
        """ produces a matlib.pyplot plot showing the losses during training.

            Args:
            ylim    : [ymin,ymax] values for the plot

            Note:
            To see the plot you may call this method from IPython / jupyter notebook.
        """
        episodes_per_value = self._training_duration.num_episodes_per_iteration
        self._plot_episodes(yvalues=self.training_losses, episodes_per_value=episodes_per_value, ylabel='losses', start_at_0=False, ylim=ylim)

    def _plot_episodes(self, yvalues, episodes_per_value: int, ylabel : str, start_at_0 : bool = True, ylim=None):
        """ yields a plot.

            Args:
            clip_stddev : if != 0 the y-axes is clipped at average +/- clip_stddev*stddev
        """
        value_count = len(yvalues)
        steps = range(0, value_count * episodes_per_value, episodes_per_value)
        if not start_at_0:
            steps = range(episodes_per_value, (value_count+1)*episodes_per_value, episodes_per_value)
        plt.xlim( 0, self._training_duration.num_episodes + 1 )
        if not ylim is None:
            plt.ylim(ylim)
        plt.plot(steps, yvalues)
        plt.ylabel(ylabel)
        plt.xlabel('episodes')

    def render_episodes_to_html(self, num_episodes : int = 10,
                                filepath : str = None,
                                fps : int = 20,
                                width : int = 640,
                                height : int = 480 ) -> str:
        """ renders all steps in num_episodes as a mp4 movie and embeds it in HTML for display
            in a jupyter notebook.

            The gym_env.render(mode='rgb_array') must yield an numpy.ndarray representing rgb values, 
            otherwise an exception is thrown.

            Args:
            num_episodes    : the number of episodes to render
            filepath        : the path to which the movie is written to. If None a filename is generated.
            fps             : frames per second, each frame contains the rendering of a single step
            height          : height iin pixels of the HTML rendered episodes
            width           : width in pixels of the HTML rendered episodes

            Note:
            o To see the plot you may call IPython.display.HTML( <agent>.render_episodes_to_html() ) from
              IPython / jupyter notebook.
            o code adapted from: https://colab.research.google.com/github/tensorflow/agents/blob/master/tf_agents/colabs/1_dqn_tutorial.ipynb
        """
        assert num_episodes >= 0, "num_episodes must be >= 0"  
        assert height >= 1, "height must be >= 1"  
        assert width >= 1, "width must be >= 1"  
        
        filepath = self.render_episodes_to_mp4( num_episodes=num_episodes, filepath=filepath, fps=fps )
        with open( filepath, 'rb' ) as f:
            video = f.read()
            b64 = base64.b64encode( video )
        os.remove( filepath )
        
        result = '''
        <video width="{0}" height="{1}" controls>
            <source src="data:video/mp4;base64,{2}" type="video/mp4">
        Your browser does not support the video tag.
        </video>'''.format( width, height, b64.decode() )
        return result
    
    def render_episodes_to_mp4(self, num_episodes : int = 10, filepath : str = None, fps : int = 20 ) -> str:
        """ renders all steps in num_episodes as a mp4 movie and stores it in filename.
            Returns the path to the written file.

            The gym_env.render(mode='rgb_array') must yield an numpy.ndarray representing rgb values, 
            otherwise an exception is thrown.

            Args:
            num_episodes    : the number of episodes to render
            filepath        : the path to which the movie is written to. If None a temp filepath is generated.
            fps             : frames per second

            Note:
            code adapted from: https://colab.research.google.com/github/tensorflow/agents/blob/master/tf_agents/colabs/1_dqn_tutorial.ipynb
        """
        assert num_episodes >= 0, "num_episodes must be >= 0"
        
        if filepath is None:
            filepath = self._gym_env_name 
            if filepath.startswith( EasyEnv.NAME_PREFIX ):
                filepath = filepath[ len(EasyEnv.NAME_PREFIX): ]
            filepath = os.path.join( tempfile.gettempdir(),
                                     next( tempfile._get_candidate_names() ) + "_" + filepath + ".mp4")
        with imageio.get_writer(filepath, fps=fps) as video:
            for _ in range( num_episodes ):
                self.play_episode( lambda gym_env, action, state, reward, done, info : video.append_data( self._render_image( gym_env ) ) )
        return filepath

    def _render_image( self, gym_env : gym.Env ):
        """ calls gym_env.render() and validates that it is an image (suitable for rendering a movie)
        """
        result = gym_env.render(mode='rgb_array')

        assert result is not None, "gym_env.render() yielded None"
        assert isinstance( result, np.ndarray ), "gym_env.render() did not yield a numpy.ndarray."
        return result






