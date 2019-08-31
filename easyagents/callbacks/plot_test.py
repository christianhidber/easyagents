import unittest
import easyagents
from easyagents.callbacks.duration import Fast
from easyagents.callbacks.plot import PlotLoss, PlotRewards, PlotSteps


class PlotLossTest(unittest.TestCase):

    def test_plotloss(self):
        agent = easyagents.PpoAgent("CartPole-v0")
        agent.train([Fast(), PlotLoss()])

    def test_plotrewards(self):
        agent = easyagents.PpoAgent("CartPole-v0")
        agent.train([Fast(), PlotRewards()])

    def test_multiple_subplots(self):
        agent = easyagents.PpoAgent("CartPole-v0")
        agent.train([Fast(), PlotRewards(), PlotLoss(), PlotSteps()])

if __name__ == '__main__':
    unittest.main()
