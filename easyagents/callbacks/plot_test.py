import unittest
import easyagents
from easyagents.callbacks.duration import Fast
from easyagents.callbacks.plot import PlotLoss, PlotRewards, PlotSteps


class PlotLossTest(unittest.TestCase):

    def test_plotloss(self):
        agent = easyagents.PpoAgent("CartPole-v0")
        agent.train([Fast(), PlotLoss()])

    def test_plotrewards(self):
        agent = easyagents.PpoAgent("CartPole-v0")
        agent.train([Fast(), PlotRewards()])

    def test_multiple_subplots(self):
        agent = easyagents.PpoAgent("CartPole-v0")
        agent.train([Fast(), PlotRewards(), PlotLoss(), PlotSteps()])

    def test(self):
        ppoAgent = easyagents.PpoAgent(gym_env_name='CartPole-v0', fc_layers=(500, 500, 500))
        ppoAgent.train([PlotLoss(ylim=(0.01, 100)), PlotSteps(), PlotRewards()],
                       learning_rate=0.0001,
                       num_iterations=2500, num_epochs_per_iteration=5, max_steps_per_episode=50,
                       num_iterations_between_eval=10, num_episodes_per_eval=10)


if __name__ == '__main__':
    unittest.main()
