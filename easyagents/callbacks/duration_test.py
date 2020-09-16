import unittest
from easyagents import agents
from easyagents.callbacks import duration, log


class DurationTest(unittest.TestCase):
    def test_single_episode(self):
        agent = agents.PpoAgent("CartPole-v0")
        count = log._CallbackCounts()
        agent.train([duration._SingleEpisode(), log._Callbacks(), count])
        assert count.train_iteration_begin_count == 1


if __name__ == '__main__':
    unittest.main()
