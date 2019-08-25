import unittest
import easyagents
from easyagents.callbacks.duration import SingleEpisode
from easyagents.callbacks.log import Count, LogCallbacks


class DurationTest(unittest.TestCase):
    def test_single_episode(self):
        agent = easyagents.PpoAgent("CartPole-v0")
        count=Count()
        agent.train([SingleEpisode(), LogCallbacks(), count])
        assert count.train_iteration_begin_count == 1


if __name__ == '__main__':
    unittest.main()
