import unittest
import tensorflow as tf
from easyagents.tfagents import Ppo

import logging

ppoAgent = Ppo( 'CartPole-v0' )
ppoAgent.train( num_training_episodes=5,
                num_training_episodes_per_iteration=2,
                num_eval_episodes=2)
