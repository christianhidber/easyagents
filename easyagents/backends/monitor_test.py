import unittest

import easyagents.core as core
import easyagents.backends.monitor as monitor
import easyagents.backends.noop as noop
import gym


class MonitorTest(unittest.TestCase):
    def setUp(self):
        self.env_name = 'CartPole-v0'
        monitor._MonitorEnv._monitor_total_counts[self.env_name] = monitor._MonitorTotalCounts(self.env_name)
        self.total = monitor._register_gym_monitor(self.env_name)
        self.env: monitor._MonitorEnv = monitor._get(gym.make(self.total.gym_env_name))

    def tearDown(self):
        monitor._MonitorEnv._register_backend_agent(None)
        monitor._MonitorEnv._monitor_total_counts[self.env_name] = monitor._MonitorTotalCounts(self.env_name)

    def test_create(self):
        assert self.env
        assert self.env.total.instances_created == 1
        assert self.env.gym_env_name == self.env_name

    def test_register_gym_monitor(self):
        assert self.env_name == self.total._original_env_name
        assert self.total.gym_env_name == monitor._MonitorEnv._NAME_PREFIX + self.env_name
        assert self.total is monitor._MonitorEnv._monitor_total_counts[self.env_name]

    def test_reset_beforeSteps_noEpisodeInc(self):
        self.env.reset()
        self.env.reset()
        assert self.total.episodes_done == 0
        assert self.env.episodes_done == 0

    def test_setbackendagent_twice(self):
        model_config = core.ModelConfig(self.env_name)
        agent = noop.BackendAgent(model_config)
        monitor._MonitorEnv._register_backend_agent(agent)
        monitor._MonitorEnv._register_backend_agent(agent)
        monitor._MonitorEnv._register_backend_agent(None)

    def test_setUp(self):
        assert self.env_name
        assert self.total
        assert self.env


if __name__ == '__main__':
    unittest.main()
