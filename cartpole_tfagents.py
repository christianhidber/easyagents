#
# simple easyagents demo using cartpole

from easyagents.tfagents import Ppo
import easyagents.logenv
import logging

logging.basicConfig(level=logging.DEBUG)
logging.info("starting")
logname = easyagents.logenv.register('CartPole-v0')
ppoAgent = Ppo( logname )
returns = ppoAgent.train( num_training_iterations=3,
                          num_training_episodes_per_iteration=3, 
                          num_training_epochs_per_iteration=3,
                          num_training_steps_in_replay_buffer=1000,
                          num_training_iterations_between_eval=1,
                          num_eval_episodes=1,
                          learning_rate=1e-4 )
logging.info("completed")

input("press enter to terminate...")
