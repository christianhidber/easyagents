import unittest

from easyagents import core, env
from easyagents.backends import tforce
from easyagents.callbacks import duration, log

import easyagents

easyagents.agents.seed = 0

class TensorforceAgentTest(unittest.TestCase):

    def setUp(self):
        self.env_name = env._StepCountEnv.register_with_gym()

    def test_dqn_train(self):
        model_config = core.ModelConfig("CartPole-v0",fc_layers=(100,))
        tc : core.DqnTrainContext = core.DqnTrainContext()
        tc.num_iterations=20000
        tc.num_steps_buffer_preload = 1000
        tc.num_iterations_between_eval=1000
        tc.max_steps_per_episode=200
        dqnAgent = tforce.TforceDqnAgent(model_config=model_config)
        dqnAgent.train(train_context=tc, callbacks=[log.Iteration(eval_only=True), log.Agent()])

    def test_dueling_dqn_train(self):
        model_config = core.ModelConfig("CartPole-v0",fc_layers=(100,))
        tc : core.DqnTrainContext = core.DqnTrainContext()
        tc.num_iterations=20000
        tc.num_steps_buffer_preload = 1000
        tc.num_iterations_between_eval=1000
        tc.max_steps_per_episode=200
        dqnAgent = tforce.TforceDuelingDqnAgent(model_config=model_config)
        dqnAgent.train(train_context=tc, callbacks=[log.Iteration(eval_only=True), log.Agent()])

    def test_ppo_train(self):
        model_config = core.ModelConfig("CartPole-v0")
        tc = core.ActorCriticTrainContext()
        ppoAgent = tforce.TforcePpoAgent(model_config=model_config)
        ppoAgent.train(train_context=tc, callbacks=[log.Iteration(), log.Agent(), duration.Fast()])

    def test_random_train(self):
        model_config = core.ModelConfig("CartPole-v0")
        tc = core.TrainContext()
        randomAgent = tforce.TforceRandomAgent(model_config=model_config)
        randomAgent.train(train_context=tc, callbacks=[log.Iteration(), log.Agent(), duration.Fast()])

    def test_reinforce_train(self):
        model_config = core.ModelConfig("CartPole-v0")
        tc = core.EpisodesTrainContext()
        reinforceAgent = tforce.TforceReinforceAgent(model_config=model_config)
        reinforceAgent.train(train_context=tc, callbacks=[log.Iteration(), log.Agent(), duration.Fast()])


if __name__ == '__main__':
    unittest.main()
