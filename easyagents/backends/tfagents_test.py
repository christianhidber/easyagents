import unittest

from easyagents import core, env
from easyagents.backends import tfagents
from easyagents.callbacks import duration, log


class TfDqnAgentTest(unittest.TestCase):

    def setUp(self):
        self.env_name = env._StepCountEnv.register_with_gym()

    def test_train(self):
        model_config = core.ModelConfig("CartPole-v0")
        tc = core.DqnTrainContext()
        dqnAgent = tfagents.TfDqnAgent(model_config=model_config)
        dqnAgent.train(train_context=tc, callbacks=[duration._SingleEpisode(), log.Iteration()])


class TfPpoAgentTest(unittest.TestCase):

    def setUp(self):
        self.env_name = env._StepCountEnv.register_with_gym()

    def test_train(self):
        model_config = core.ModelConfig("CartPole-v0")
        tc = core.ActorCriticTrainContext()
        ppoAgent = tfagents.TfPpoAgent(model_config=model_config)
        ppoAgent.train(train_context=tc, callbacks=[duration.Fast(), log.Iteration()])

class TfRandomAgentTest(unittest.TestCase):

    def setUp(self):
        self.env_name = env._StepCountEnv.register_with_gym()

    def test_train(self):
        model_config = core.ModelConfig("CartPole-v0")
        tc = core.TrainContext()
        randomAgent = tfagents.TfRandomAgent(model_config=model_config)
        randomAgent.train(train_context=tc, callbacks=[duration.Fast(), log.Iteration()])
        assert tc.episodes_done_in_iteration == 1

"""
    def test_play(self):
        model_config = core.ModelConfig("CartPole-v0")
        randomAgent = tfagents.TfRandomAgent(model_config=model_config)
        pc=core.PlayContext()
        pc.max_steps_per_episode=10
        randomAgent.play(play_context=pc,callbacks=[])
        assert pc.num_episodes == 1
        """


if __name__ == '__main__':
    unittest.main()
