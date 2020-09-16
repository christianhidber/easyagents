import unittest
import os
import tempfile

import easyagents
from easyagents import agents
from easyagents.callbacks import save, duration

_line_world_name = easyagents.env._LineWorldEnv.register_with_gym()


class SaveTest(unittest.TestCase):

    def test_directory(self):
        best = save.Best()
        assert best.directory
        temp_path = os.path.abspath(os.path.join(tempfile.gettempdir(), "save_test"))
        best = save.Best(temp_path)
        assert temp_path == best.directory

    def test_best(self):
        agent = agents.PpoAgent(_line_world_name)
        best = save.Best()
        agent.train([best],
                    num_iterations_between_eval=1, num_episodes_per_iteration=10, num_iterations=3,
                    default_plots=False)
        os.path.isdir(best.directory)
        assert len(best.saved_agents) > 0
        (episode, reward, dir) = best.saved_agents[0]
        os.path.isdir(dir)
        agent2 = easyagents.agents.load(dir)
        assert agent2
        agent2.evaluate()

    def test_every(self):
        agent = agents.PpoAgent(_line_world_name)
        every = save.Every(num_evals_between_save=1)
        agent.train([every],
                    num_iterations_between_eval=1, num_episodes_per_iteration=10, num_iterations=3,
                    default_plots=False)
        os.path.isdir(every.directory)
        assert len(every.saved_agents) == 4
        for (episode, reward, dir) in every.saved_agents:
            os.path.isdir(dir)


if __name__ == '__main__':
    unittest.main()
