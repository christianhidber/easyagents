import unittest

import easyagents
from easyagents.callbacks.duration import Fast as Fast
from easyagents.callbacks.log import *


class LogCallbacksTest(unittest.TestCase):

    def test_log_callbacks(self):
        agent = easyagents.PpoAgent("CartPole-v0")
        agent.train([LogCallbacks(), Fast()])

    def test_log_loss(self):
        agent = easyagents.PpoAgent("CartPole-v0")
        agent.train([LogLoss(), Fast()])
