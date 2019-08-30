import unittest
import easyagents
from easyagents.callbacks.duration import Fast
from easyagents.callbacks.plot import PlotLoss, PlotRewards


class PlotLossTest(unittest.TestCase):

    def test_plotloss(self):
        agent = easyagents.PpoAgent("CartPole-v0")
        agent.train([Fast(), PlotLoss()])

    def test_plotrewards(self):
        agent = easyagents.PpoAgent("CartPole-v0")
        agent.train([Fast(), PlotRewards()])

if __name__ == '__main__':
    unittest.main()
