import pytest
import unittest
import logging
import os
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

def get_backends(agent: Optional[Type[EasyAgent]] = None, skip_v1: bool = False):
    result = [b for b in agents.get_backends(agent, skip_v1) if b != 'default']
    return result

def max_avg_rewards(tc: core.TrainContext):
    max_avg_rewards = max([avg_rewards for (min_rewards, avg_rewards, max_rewards) in tc.eval_rewards.values()])
    return max_avg_rewards

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
#        assert 'tensorforce' in backends

    def test_getbackends_randomagent(self):
        assert agents._backends is not None
        backends = agents.get_backends(agents.RandomAgent)
        assert 'default' in backends
        assert 'tfagents' in backends
#        assert 'tensorforce' in backends

    def test_prepare_callbacks(self):
        agent = agents.PpoAgent(_line_world_name)
        c = [plot.ToMovie(), plot.Rewards()]
        d = agent._prepare_callbacks(c, default_plots=None, default_plot_callbacks=[])
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
        b2 = easyagents.backends.default.BackendAgentFactory()
        old_length = len(easyagents.agents.get_backends())
        assert b2 not in agents._backends
        agents.register_backend(backend=b2)
        assert old_length == len(easyagents.agents.get_backends())
        assert b2 in agents._backends

    def test_register_backend_None_exception(self):
        with pytest.raises(AssertionError):
            agents.register_backend(backend=None)


class CemAgentTest(unittest.TestCase):

    def train_and_assert(self, agent_type, is_v1: bool, num_iterations=100):
        logger = logging.warning
        v2_backends = [b for b in get_backends(agent_type, skip_v1=True)]
        v1_backends = [b for b in get_backends(agent_type) if (b not in v2_backends)]
        backends = v1_backends if is_v1 else v2_backends
        for backend in backends:
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

    @pytest.mark.skipif(easyagents.backends.core._tensorflow_v2_eager_enabled, reason="tfv2 active")
    @pytest.mark.tfv1
    def test_cem_v1(self):
        self.train_and_assert(CemAgent, True)

    def test_create(self):
        with pytest.raises(Exception) as e_info:
            CemAgent('LineWorld-v0', fc_layers=(100,))


class DqnAgentsTest(unittest.TestCase):

    def train_and_eval(self, agent_type, backend, num_iterations):
        dqn_agent: DqnAgent = agent_type(_line_world_name, fc_layers=(100,), backend=backend)
        tc: core.TrainContext = dqn_agent.train([log.Duration(), log.Iteration(eval_only=True), log.Agent()],
                                                num_iterations=num_iterations,
                                                num_steps_buffer_preload=1000,
                                                num_iterations_between_eval=500,
                                                max_steps_per_episode=200,
                                                default_plots=False)
        return max_avg_rewards(tc)

    def train_and_assert(self, agent_type, is_v1: bool, num_iterations=1000):
        logger = logging.warning
        v2_backends = [b for b in get_backends(agent_type, skip_v1=True) if b != 'default']
        v1_backends = [b for b in get_backends(agent_type) if (b not in v2_backends) and b != 'default']
        backends = v1_backends if is_v1 else v2_backends
        for backend in backends:
            current_num_iterations = num_iterations
            if backend == 'tensorforce':
                current_num_iterations = num_iterations * 3
            logger(f'backend={backend} agent={agent_type}, num_iterations={current_num_iterations}')
            max_avg_rewards = self.train_and_eval(agent_type=agent_type, backend=backend,
                                                num_iterations=current_num_iterations)
            assert max_avg_rewards >= 20, f'agent_type={agent_type} backend={backend} num_iterations={num_iterations}'

    @pytest.mark.skipif(easyagents.backends.core._tensorflow_v2_eager_enabled, reason="tfv2 active")
    @pytest.mark.tfv1
    def test_dqn_v1(self):
        self.train_and_assert(DqnAgent, True)

    def test_dqn_v2(self):
        self.train_and_assert(DqnAgent, False)

    @pytest.mark.skipif(easyagents.backends.core._tensorflow_v2_eager_enabled, reason="tfv2 active")
    @pytest.mark.tfv1
    def test_double_dqn_v1(self):
        self.train_and_assert(DoubleDqnAgent, True)

    def test_double_dqn_v2(self):
        self.train_and_assert(DoubleDqnAgent, False)

    @pytest.mark.skipif(easyagents.backends.core._tensorflow_v2_eager_enabled, reason="tfv2 active")
    @pytest.mark.tfv1
    def test_dueling_dqn_v1(self):
        self.train_and_assert(DuelingDqnAgent, True)

    def test_dueling_dqn_v2(self):
        self.train_and_assert(DuelingDqnAgent, False)


# noinspection PyTypeChecker
class PpoAgentTest(unittest.TestCase):

    def test_callback_single(self):
        for backend in get_backends(PpoAgent):
            env._StepCountEnv.clear()
            agent = PpoAgent(_step_count_name, backend=backend)
            agent.train(duration._SingleEpisode())
            assert env._StepCountEnv.reset_count <= 2

    def test_train_CartPole(self):
        agents.seed=0
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
            ppo = agents.PpoAgent(gym_env_name=_step_count_name, backend=backend)
            ppo.load(temp_dir)
            ppo.play(default_plots=False, num_episodes=1, callbacks=[])
            os.rmdir(temp_dir)


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


class SacAgentTest(unittest.TestCase):

    def test_train(self):
        for backend in get_backends(SacAgent):
            sac_agent: SacAgent = SacAgent(_mountaincart_continuous_name, backend=backend)
            tc : core.TrainContext = sac_agent.train([log.Duration(), log.Iteration(eval_only=True)],
                                                        default_plots=False)
            r = max_avg_rewards(tc)
            assert r >= -1


class EasyAgentTest(unittest.TestCase):

    def assert_properties_for_metric(self, metric, num_episodes):
        assert metric.min <= metric.max
        assert metric.mean <= metric.max
        assert metric.mean >= metric.min
        assert metric.std >= 0
        assert len(metric.all) == num_episodes

    def test_evaluate(self):
        random_agent = RandomAgent(_line_world_name)
        num_episodes = 5
        metrics = random_agent.evaluate(num_episodes=num_episodes)
        self.assert_properties_for_metric(metrics.steps, num_episodes)
        self.assert_properties_for_metric(metrics.rewards, num_episodes)


if __name__ == '__main__':
    unittest.main()
