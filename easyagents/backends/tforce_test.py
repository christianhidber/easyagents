import unittest

from easyagents import core, env
from easyagents.backends import tforce
from easyagents.backends import core as bcore
from easyagents.callbacks import log

import easyagents

easyagents.agents.seed = 0
_lineworld_name = env._LineWorldEnv.register_with_gym()
_cartpole_name = 'CartPole-v0'


class TensorforceAgentTest(unittest.TestCase):

    def setUp(self):
        self.env_name = env._StepCountEnv.register_with_gym()

    def test_dqn_train(self):
        model_config = core.ModelConfig(_cartpole_name, fc_layers=(100,100))
        tc: core.StepsTrainContext = core.StepsTrainContext()
        tc.num_iterations = 10000
        tc.num_steps_buffer_preload = 500
        tc.num_iterations_between_eval = 500
        tc.max_steps_per_episode = 200
        dqnAgent = tforce.TforceDqnAgent(model_config=model_config)
        dqnAgent.train(train_context=tc, callbacks=[log.Iteration(eval_only=True), log.Agent()])
        (min_r,avg_r,max_r) = tc.eval_rewards[tc.episodes_done_in_training]
        assert avg_r > 100

    def test_dueling_dqn_train(self):
        model_config = core.ModelConfig(_cartpole_name, fc_layers=(100,))
        tc: core.StepsTrainContext = core.StepsTrainContext()
        tc.num_iterations = 2000
        tc.num_steps_buffer_preload = 100
        tc.num_iterations_between_eval = 100
        tc.max_steps_per_episode = 200
        dqnAgent = tforce.TforceDuelingDqnAgent(model_config=model_config)
        dqnAgent.train(train_context=tc, callbacks=[log.Iteration(eval_only=True), log.Agent()])

    def test_ppo_train(self):
        model_config = core.ModelConfig(_cartpole_name)
        tc = core.PpoTrainContext()
        tc.num_iterations=20
        ppoAgent = tforce.TforcePpoAgent(model_config=model_config)
        ppoAgent.train(train_context=tc, callbacks=[log.Iteration(), log.Agent()])
        (min_r,avg_r,max_r) = tc.eval_rewards[tc.episodes_done_in_training]
        assert avg_r > 100

    def test_random_train(self):
        model_config = core.ModelConfig(_cartpole_name)
        tc = core.TrainContext()
        tc.num_iterations=50
        randomAgent = tforce.TforceRandomAgent(model_config=model_config)
        randomAgent.train(train_context=tc, callbacks=[log.Iteration(), log.Agent()])
        (min_r,avg_r,max_r) = tc.eval_rewards[tc.episodes_done_in_training]
        assert avg_r < 50

    def test_reinforce_train(self):
        model_config = core.ModelConfig(_cartpole_name)
        tc = core.EpisodesTrainContext()
        tc.num_iterations=50
        reinforceAgent = tforce.TforceReinforceAgent(model_config=model_config)
        reinforceAgent.train(train_context=tc, callbacks=[log.Iteration(), log.Agent()])
        (min_r,avg_r,max_r) = tc.eval_rewards[tc.episodes_done_in_training]
        assert avg_r > 100

    def test_save_load(self):
        model_config = core.ModelConfig(_cartpole_name)
        tc = core.PpoTrainContext()
        tc.num_iterations=20
        ppo_agent = tforce.TforcePpoAgent(model_config=model_config)
        ppo_agent.train(train_context=tc, callbacks=[log.Iteration(), log.Agent()])
        tempdir = bcore._get_temp_path()
        bcore._mkdir(tempdir)
        ppo_agent.save(tempdir, [])
        ppo_agent = tforce.TforcePpoAgent(model_config=model_config)
        ppo_agent.load(tempdir, [])
        pc = core.PlayContext()
        pc.max_steps_per_episode = 10
        pc.num_episodes = 3
        ppo_agent.play(play_context=pc, callbacks=[])
        (min_r,avg_r,max_r) = pc.rewards[pc.episodes_done]
        assert avg_r > 100
        bcore._rmpath(tempdir)


if __name__ == '__main__':
    unittest.main()
