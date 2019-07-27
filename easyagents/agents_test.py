import unittest

from easyagents.config import LoggingMinimal
from easyagents.config import TrainingFast
from easyagents.config import TrainingSingleEpisode
from easyagents.tfagents import PpoAgent


class TestAgents(unittest.TestCase):

    def setUp(self):
        self.ppo = PpoAgent('CartPole-v0', training=TrainingSingleEpisode(), logging=LoggingMinimal())
        self.ppo.train()

    def test_render_episodes_to_mp4(self):
        self.ppo.render_episodes_to_mp4()

    def test_plot_no_ylim(self):
        self.ppo.plot_average_rewards()

    def test_plot_with_ylim(self):
        self.ppo.plot_average_rewards(ylim=[10, 20])

    def test_plot_episodes_scale_success(self):
        self.ppo.plot_average_rewards(scale=['log','linear','linear'])

    def test_plot_episodes_scale_exception(self):
        self.assertRaises(BaseException, lambda : self.ppo.plot_episodes(scale='log'))

    def test_plot_episodes_ylim_success(self):
        self.ppo.plot_episodes(ylim=[(-1000, 10000), (0, 100), (0, 200)])

    def test_plot_episodes_ylim_exception(self):
        self.assertRaises(BaseException, lambda : self.ppo.plot_episodes(ylim=[(-1000, 10000), (0, 100)]))

    def test_render_episodes(self):
        self.ppo.render_episodes()


if __name__ == '__main__':
    unittest.main()
