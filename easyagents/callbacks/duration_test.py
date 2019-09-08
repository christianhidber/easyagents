import unittest
import easyagents
from easyagents.callbacks.duration import _SingleEpisode
from easyagents.callbacks.log import _CallbackCounts, _Callbacks


class DurationTest(unittest.TestCase):
    def test_single_episode(self):
        agent = easyagents.PpoAgent("CartPole-v0")
        count=_CallbackCounts()
        agent.train([_SingleEpisode(), _Callbacks(), count])
        assert count.train_iteration_begin_count == 1


if __name__ == '__main__':
    unittest.main()
