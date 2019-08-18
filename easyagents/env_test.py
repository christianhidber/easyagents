import unittest
from easyagents.env import _is_registered_with_gym


class EnvTest(unittest.TestCase):
    def test_is_registered_with_gym(self):
        assert _is_registered_with_gym('CartPole-v0')

    def test_is_registered_with_gym_missingRegistration(self):
        assert not _is_registered_with_gym('MyEnv-v0')


if __name__ == '__main__':
    unittest.main()
