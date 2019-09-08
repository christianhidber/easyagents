import unittest
import tempfile
import os

import easyagents

from easyagents.callbacks import duration, plot


class PlotTest(unittest.TestCase):

    def test_play_plotstate(self):
        agent = easyagents.PpoAgent("CartPole-v0")
        agent.train([duration._SingleEpisode()])
        agent.play([plot.State()])

    def test_play_plotrewards(self):
        agent = easyagents.PpoAgent("CartPole-v0")
        agent.train([duration._SingleIteration()])
        agent.play([plot.Rewards()])

    def test_train_plotloss(self):
        agent = easyagents.PpoAgent("CartPole-v0")
        agent.train([duration._SingleIteration(), plot.Loss()])

    def test_train_plotrewards(self):
        agent = easyagents.PpoAgent("CartPole-v0")
        agent.train([duration._SingleIteration(), plot.Rewards()])

    def test_train_plotstate(self):
        agent = easyagents.PpoAgent("CartPole-v0")
        agent.train([duration._SingleIteration(), plot.State()])

    def test_train_plotsteps(self):
        agent = easyagents.PpoAgent("CartPole-v0")
        agent.train([duration._SingleIteration(), plot.Steps()])

    def test_train_multiple_subplots(self):
        agent = easyagents.PpoAgent("CartPole-v0")
        agent.train([duration._SingleIteration(), plot.State(), plot.Rewards(), plot.Loss(), plot.Steps()])

    def test_train_tomovie(self):
        agent = easyagents.PpoAgent("CartPole-v0")
        agent.train([duration._SingleIteration(), plot.Rewards(), plot.ToMovie()])

    def test_train_tomovie_with_filename(self):
        f = tempfile.NamedTemporaryFile(delete=False)
        filepath = f.name
        f.close()
        os.remove(filepath)
        assert not os.path.isfile(filepath)
        agent = easyagents.PpoAgent("CartPole-v0")
        m = plot.ToMovie(filepath=filepath, fps=10)
        agent.train([duration._SingleIteration(), plot.Rewards(), m])
        os.remove(m.filepath)



if __name__ == '__main__':
    unittest.main()
