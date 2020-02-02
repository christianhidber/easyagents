import pytest
import unittest
import logging
from typing import Optional, Type

import easyagents
from easyagents.agents import CemAgent, DqnAgent, DoubleDqnAgent, DuelingDqnAgent, PpoAgent, \
    RandomAgent, ReinforceAgent, SacAgent, EasyAgent
from easyagents import env, core, agents
from easyagents.callbacks import duration, log, plot
import easyagents.backends.core

_step_count_name = env._StepCountEnv.register_with_gym()
_line_world_name = env._LineWorldEnv.register_with_gym()
_cartpole_name = "CartPole-v0"
_mountaincart_continuous_name = "MountainCarContinuous-v0"
easyagents.agents.seed = 0


def get_backends(agent: Optional[Type[EasyAgent]] = None):
    result = [b for b in agents.get_backends(agent) if b != 'default']
    assert result, f'no backend found for agent {agent}.'
    return result


def max_avg_rewards(tc: core.TrainContext):
    result = max([avg_rewards for (min_rewards, avg_rewards, max_rewards) in tc.eval_rewards.values()])
    return result


# noinspection PyTypeChecker
class BackendRegistrationTest(unittest.TestCase):
    class MyBackend(easyagents.backends.core.BackendAgentFactory):
        backend_name = "MyBackend"

    def setUp(self):
        self._oldbackends = agents._backends.copy()

    def tearDown(self):
        agents._backends = self._oldbackends

    def test_getbackends(self):
        assert agents._backends is not None
        assert agents.get_backends() is not None

    def test_getbackends_ppoagent(self):
        assert agents._backends is not None
        backends = agents.get_backends(agents.PpoAgent)
        assert 'default' in backends
        assert 'tfagents' in backends

    def test_getbackends_randomagent(self):
        assert agents._backends is not None
        backends = agents.get_backends(agents.RandomAgent)
        assert 'default' in backends
        assert 'tfagents' in backends

    def test_prepare_callbacks(self):
        agent = agents.PpoAgent(_line_world_name)
        c = [plot.ToMovie(), plot.Rewards()]
        d = agent._add_plot_callbacks(c, default_plots=None, default_plot_callbacks=[])
        assert isinstance(d[0], plot._PreProcess)
        assert isinstance(d[1], plot.Rewards)
        assert isinstance(d[-2], plot._PostProcess)
        assert isinstance(d[-1], plot.ToMovie)

    def test_register(self):
        old_len = len(easyagents.agents.get_backends())
        agents.register_backend(BackendRegistrationTest.MyBackend())
        b = easyagents.agents.get_backends()
        assert (old_len + 1) == len(b)
        assert BackendRegistrationTest.MyBackend.backend_name in b

    def test_register_backend_twice(self):
        b2 = easyagents.backends.default.DefaultAgentFactory(register_tensorforce=False)
        old_length = len(easyagents.agents.get_backends())
        assert b2 not in agents._backends
        agents.register_backend(backend=b2)
        assert old_length == len(easyagents.agents.get_backends())
        assert b2 in agents._backends

    def test_register_backend_None_exception(self):
        with pytest.raises(AssertionError):
            agents.register_backend(backend=None)


class CemAgentTest(unittest.TestCase):

    def train_and_assert(self, agent_type, num_iterations=100):
        logger = logging.warning
        for backend in get_backends(agent_type):
            logger(f'backend={backend} agent={agent_type}, num_iterations={num_iterations}')
            cem_agent: CemAgent = agent_type('CartPole-v0', fc_layers=(100,), backend=backend)
            tc: core.TrainContext = cem_agent.train([log.Duration(), log.Iteration(eval_only=True), log.Agent()],
                                                    num_iterations=num_iterations,
                                                    num_iterations_between_eval=10,
                                                    max_steps_per_episode=200,
                                                    default_plots=False)
            (min_steps, avg_steps, max_steps) = tc.eval_steps[tc.episodes_done_in_training]
            assert max_steps >= 100
            assert avg_steps >= 50

    def test_create(self):
        with pytest.raises(Exception):
            CemAgent('LineWorld-v0', fc_layers=(100,))


class DqnAgentsTest(unittest.TestCase):

    def train_and_eval(self, agent_type, backend, num_iterations):
        dqn_agent: DqnAgent = agent_type(_cartpole_name, fc_layers=(100,), backend=backend)
        tc: core.TrainContext = dqn_agent.train([log.Duration(), log.Iteration(eval_only=True), log.Agent()],
                                                num_iterations=num_iterations,
                                                num_steps_buffer_preload=1000,
                                                num_iterations_between_eval=500,
                                                max_steps_per_episode=200,
                                                default_plots=False)
        return max_avg_rewards(tc)

    def train_and_assert(self, agent_type, num_iterations=1000):
        logger = logging.warning
        for backend in get_backends(agent_type):
            current_num_iterations = num_iterations
            if backend == 'tensorforce':
                current_num_iterations = num_iterations * 3
            logger(f'backend={backend} agent={agent_type}, num_iterations={current_num_iterations}')
            rewards = self.train_and_eval(agent_type=agent_type, backend=backend, num_iterations=current_num_iterations)
            assert rewards >= 20, f'agent_type={agent_type} backend={backend} num_iterations={num_iterations}'

    def test_dqn(self):
        agents._activate_tfagents()
        self.train_and_assert(DqnAgent)

    @pytest.mark.skipif(easyagents.backends.core._tf_eager_execution_active, reason="_tf_eager_execution_active")
    @pytest.mark.tforce
    def test_dqn_tforce(self):
        agents.activate_tensorforce()
        self.train_and_assert(DqnAgent)

    def test_double_dqn(self):
        agents._activate_tfagents()
        with pytest.raises(Exception):
            self.train_and_assert(DoubleDqnAgent)

    @pytest.mark.skipif(easyagents.backends.core._tf_eager_execution_active, reason="_tf_eager_execution_active")
    @pytest.mark.tforce
    def test_dueling_dqn_tforce(self):
        agents.activate_tensorforce()
        self.train_and_assert(DuelingDqnAgent)



class EasyAgentTest(unittest.TestCase):

    def assert_properties_for_metric(self, metric, num_episodes):
        assert metric.min <= metric.max
        assert metric.mean <= metric.max
        assert metric.mean >= metric.min
        assert metric.std >= 0
        assert len(metric.all) == num_episodes

    def test_agent_saver_set(self):
        random_agent = RandomAgent(_line_world_name)
        assert random_agent._backend_agent._agent_context._agent_saver

    def test_evaluate(self):
        random_agent = RandomAgent(_line_world_name)
        num_episodes = 5
        metrics = random_agent.evaluate(num_episodes=num_episodes)
        self.assert_properties_for_metric(metrics.steps, num_episodes)
        self.assert_properties_for_metric(metrics.rewards, num_episodes)

    def test_save_no_trained_policy_exception(self):
        p1 = PpoAgent(gym_env_name=_line_world_name, fc_layers=(10, 20, 30), backend='tfagents')
        with pytest.raises(Exception):
            p1.save()

    def test_save_load_play(self):
        oldseed = agents.seed
        agents.seed = 123
        p1 = PpoAgent(gym_env_name=_line_world_name, fc_layers=(10, 20, 30), backend='tfagents')
        p1.train(callbacks=[duration._SingleEpisode()], default_plots=False)
        d = p1.save()
        agents.seed = oldseed
        p2: EasyAgent = agents.load(d)
        self.assert_are_equal(p1, p2)
        p2.play(default_plots=False, num_episodes=1)

    def test_seed(self):
        oldseed = agents.seed
        agents.seed = 123
        random_agent = RandomAgent(_line_world_name)
        assert random_agent._model_config.seed == 123
        agents.seed = oldseed

    def test_to_dict_from_dict(self):
        oldseed = agents.seed
        agents.seed = 123
        p1 = PpoAgent(gym_env_name=_line_world_name, fc_layers=(10, 20, 30), backend='tfagents')
        d = p1._to_dict()
        agents.seed = oldseed
        p2: EasyAgent = EasyAgent._from_dict(d)
        self.assert_are_equal(p1, p2)

    def assert_are_equal(self, p1: EasyAgent, p2: EasyAgent):
        assert p1
        assert p2
        assert p2._backend_name == p1._backend_name
        assert p2._backend_agent.__class__ == p1._backend_agent.__class__
        assert p2._model_config.original_env_name == p1._model_config.original_env_name
        assert p2._model_config.seed == p1._model_config.seed
        assert p2._model_config.gym_env_name == p1._model_config.gym_env_name
        assert p2._model_config.fc_layers == p1._model_config.fc_layers


# noinspection PyTypeChecker
class PpoAgentTest(unittest.TestCase):

    def test_callback_single(self):
        for backend in get_backends(PpoAgent):
            env._StepCountEnv.clear()
            agent = PpoAgent(_step_count_name, backend=backend)
            agent.train(duration._SingleEpisode())
            assert env._StepCountEnv.reset_count <= 2

    def test_train(self):
        agents.seed = 0
        for backend in get_backends(PpoAgent):
            ppo = PpoAgent(gym_env_name=_cartpole_name, backend=backend)
            tc = core.PpoTrainContext()
            tc.num_iterations = 10
            tc.num_episodes_per_iteration = 10
            tc.max_steps_per_episode = 200
            tc.num_epochs_per_iteration = 5
            tc.num_iterations_between_eval = 5
            tc.num_episodes_per_eval = 5
            ppo.train([log.Iteration()], train_context=tc, default_plots=False)
            assert max_avg_rewards(tc) >= 50

    @pytest.mark.skipif(easyagents.backends.core._tf_eager_execution_active, reason="_tf_eager_execution_active")
    @pytest.mark.tforce
    def test_train_tforce(self):
        agents.activate_tensorforce()
        self.test_train()

    def test_train_single_episode(self):
        for backend in get_backends(PpoAgent):
            ppo = agents.PpoAgent(gym_env_name=_step_count_name, backend=backend)
            count = log._CallbackCounts()
            ppo.train([log.Agent(), count, duration._SingleEpisode()])
            assert count.gym_init_begin_count == count.gym_init_end_count == 1
            assert count.gym_step_begin_count == count.gym_step_end_count
            assert count.gym_step_begin_count < 10 + count.gym_reset_begin_count

    def test_play_single_episode(self):
        for backend in get_backends(PpoAgent):
            ppo = agents.PpoAgent(gym_env_name=_step_count_name, backend=backend)
            count = log._CallbackCounts()
            cb = [log.Agent(), count, duration._SingleEpisode()]
            ppo.train(duration._SingleEpisode())
            ppo.play(cb)
            assert count.gym_init_begin_count == count.gym_init_end_count == 1
            assert count.gym_step_begin_count == count.gym_step_end_count <= 10

    def test_save_load(self):
        for backend in get_backends(PpoAgent):
            ppo = agents.PpoAgent(gym_env_name=_step_count_name, backend=backend)
            ppo.train([duration._SingleEpisode()], default_plots=False)
            temp_dir = ppo.save()
            ppo = agents.load(temp_dir)
            ppo.play(default_plots=False, num_episodes=1, callbacks=[])
            easyagents.backends.core._rmpath(temp_dir)

    @pytest.mark.skipif(easyagents.backends.core._tf_eager_execution_active, reason="_tf_eager_execution_active")
    @pytest.mark.tforce
    def test_save_load_tforce(self):
        agents.activate_tensorforce()
        self.test_save_load()


class RandomAgentTest(unittest.TestCase):

    def test_train(self):
        for backend in get_backends(RandomAgent):
            random_agent = RandomAgent(_line_world_name, backend=backend)
            tc: core.TrainContext = random_agent.train([log.Duration(), log.Iteration()],
                                                       num_iterations=10,
                                                       max_steps_per_episode=100,
                                                       default_plots=False)
            r = max_avg_rewards(tc)
            assert r >= 0

    @pytest.mark.skipif(easyagents.backends.core._tf_eager_execution_active, reason="_tf_eager_execution_active")
    @pytest.mark.tforce
    def test_train_tforce(self):
        agents.activate_tensorforce()
        self.test_train()


class ReinforceAgentTest(unittest.TestCase):

    def test_train(self):
        for backend in get_backends(RandomAgent):
            reinforce_agent: ReinforceAgent = ReinforceAgent(_line_world_name, backend=backend)
            tc: core.TrainContext = reinforce_agent.train([log.Duration(), log.Iteration()],
                                                          num_iterations=10,
                                                          max_steps_per_episode=200,
                                                          default_plots=False)
            r = max_avg_rewards(tc)
            assert r >= 5

    @pytest.mark.skipif(easyagents.backends.core._tf_eager_execution_active, reason="_tf_eager_execution_active")
    @pytest.mark.tforce
    def test_train_tforce(self):
        agents.activate_tensorforce()
        self.test_train()


class SacAgentTest(unittest.TestCase):

    def test_train(self):
        for backend in get_backends(SacAgent):
            sac_agent: SacAgent = SacAgent(_mountaincart_continuous_name, backend=backend)
            tc: core.TrainContext = sac_agent.train([log.Duration(), log.Iteration(eval_only=True), duration.Fast()],
                                                    default_plots=False)
            r = max_avg_rewards(tc)
            assert r >= -1


if __name__ == '__main__':
    unittest.main()
