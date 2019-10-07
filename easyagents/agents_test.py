import pytest
import unittest
import logging

from easyagents.agents import CemAgent, ReinforceAgent, PpoAgent, DqnAgent, DoubleDqnAgent, DuelingDqnAgent, RandomAgent
from easyagents.agents import get_backends
from easyagents import env, core, agents
from easyagents.callbacks import duration, log, plot
import easyagents.backends.core

_env_name = env._StepCountEnv.register_with_gym()
easyagents.agents.seed = 0


# noinspection PyTypeChecker
class BackendRegistrationTest(unittest.TestCase):
    class MyBackend(easyagents.backends.core.BackendAgentFactory):
        name = "MyBackend"

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
        assert 'tensorforce' in backends

    def test_getbackends_randomagent(self):
        assert agents._backends is not None
        backends = agents.get_backends(agents.RandomAgent)
        assert 'default' in backends
        assert 'tfagents' in backends
        assert 'tensorforce' not in backends

    def test_prepare_callbacks(self):
        agent = agents.PpoAgent("CartPole-v0")
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
        assert BackendRegistrationTest.MyBackend.name in b

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
        v2_backends = [b for b in get_backends(agent_type, skip_v1=True) if b != 'default']
        v1_backends = [b for b in get_backends(agent_type) if (not b in v2_backends) and  b != 'default']
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

class DqnAgentsTest(unittest.TestCase):

    def train_and_assert(self, agent_type, is_v1: bool, num_iterations=10000):
        logger = logging.warning
        v2_backends = [b for b in get_backends(agent_type, skip_v1=True) if b != 'default']
        v1_backends = [b for b in get_backends(agent_type) if (not b in v2_backends) and  b != 'default']
        backends = v1_backends if is_v1 else v2_backends
        for backend in backends:
            current_num_iterations=num_iterations
            if backend=='tensorforce':
                current_num_iterations=num_iterations*3
            logger(f'backend={backend} agent={agent_type}, num_iterations={current_num_iterations}')
            dqn_agent: DqnAgent = agent_type('CartPole-v0', fc_layers=(100,), backend=backend)
            tc: core.TrainContext = dqn_agent.train([log.Duration(), log.Iteration(eval_only=True), log.Agent()],
                                                    num_iterations=current_num_iterations,
                                                    num_steps_buffer_preload=1000,
                                                    num_iterations_between_eval=500,
                                                    max_steps_per_episode=200,
                                                    default_plots=False)
            (min_steps, avg_steps, max_steps) = tc.eval_steps[tc.episodes_done_in_training]
            assert avg_steps >= 100

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
            agent = PpoAgent(_env_name, backend=backend)
            agent.train(duration._SingleEpisode())
            assert env._StepCountEnv.reset_count <= 2

    def test_train_cartpole(self):
        for backend in get_backends(PpoAgent):
            ppo = PpoAgent(gym_env_name="CartPole-v0", backend=backend)
            tc = core.ActorCriticTrainContext()
            tc.num_iterations = 3
            tc.num_episodes_per_iteration = 10
            tc.max_steps_per_episode = 500
            tc.num_epochs_per_iteration = 5
            tc.num_iterations_between_eval = 2
            tc.num_episodes_per_eval = 5
            ppo.train([log.Iteration()], train_context=tc)

    def test_train_single_episode(self):
        for backend in get_backends(PpoAgent):
            ppo = agents.PpoAgent(gym_env_name=_env_name, backend=backend)
            count = log._CallbackCounts()
            ppo.train([log.Agent(), count, duration._SingleEpisode()])
            assert count.gym_init_begin_count == count.gym_init_end_count == 1
            assert count.gym_step_begin_count == count.gym_step_end_count
            assert count.gym_step_begin_count < 10 + count.gym_reset_begin_count

    def test_play_single_episode(self):
        for backend in get_backends(PpoAgent):
            ppo = agents.PpoAgent(gym_env_name=_env_name, backend=backend)
            count = log._CallbackCounts()
            cb = [log.Agent(), count, duration._SingleEpisode()]
            ppo.train(duration._SingleEpisode())
            ppo.play(cb)
            assert count.gym_init_begin_count == count.gym_init_end_count == 1
            assert count.gym_step_begin_count == count.gym_step_end_count <= 10


class RandomAgentTest(unittest.TestCase):

    def test_train(self):
        for backend in get_backends(RandomAgent):
            random_agent = RandomAgent('CartPole-v0', backend=backend)
            tc: core.TrainContext = random_agent.train([log.Duration(), log.Iteration()],
                                                       num_iterations=10,
                                                       max_steps_per_episode=100,
                                                       default_plots=False)
            (min_steps, avg_steps, max_steps) = tc.eval_steps[tc.episodes_done_in_training]
            assert avg_steps >= 10


class ReinforceAgentTest(unittest.TestCase):

    def test_train(self):
        for backend in get_backends(RandomAgent):
            reinforce_agent: ReinforceAgent = ReinforceAgent('CartPole-v0', backend=backend)
            tc: core.TrainContext = reinforce_agent.train([log.Duration(), log.Iteration()],
                                                          num_iterations=10,
                                                          max_steps_per_episode=200,
                                                          default_plots=False)
            (min_steps, avg_steps, max_steps) = tc.eval_steps[tc.episodes_done_in_training]
            assert avg_steps >= 10


if __name__ == '__main__':
    unittest.main()
