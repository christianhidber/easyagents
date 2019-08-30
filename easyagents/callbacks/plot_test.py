import unittest
import easyagents
from easyagents.callbacks.duration import Fast
from easyagents.callbacks.plot import PlotLoss


class PlotLossTest(unittest.TestCase):

    def test_plotloss(self):
        agent = easyagents.PpoAgent("CartPole-v0")
        agent.train([Fast(),PlotLoss()])

if __name__ == '__main__':
    unittest.main()
