import unittest
import easyagents
from easyagents.callbacks.duration import Fast, SingleEpisode
from easyagents.callbacks.plot import PlotLoss, PlotRewards, PlotSteps, PlotState


class PlotLossTest(unittest.TestCase):

    def test_play_plotstate(self):
        agent = easyagents.PpoAgent("CartPole-v0")
        agent.train([SingleEpisode()])
        agent.play([PlotState()])

    def test_play_plotrewards(self):
        agent = easyagents.PpoAgent("CartPole-v0")
        agent.train([Fast(),PlotRewards()])
        agent.play([PlotState(),PlotRewards()])

    def test_train_plotloss(self):
        agent = easyagents.PpoAgent("CartPole-v0")
        agent.train([Fast(), PlotLoss()])

    def test_train_plotrewards(self):
        agent = easyagents.PpoAgent("CartPole-v0")
        agent.train([Fast(), PlotRewards()])

    def test_train_plotstate(self):
        agent = easyagents.PpoAgent("CartPole-v0")
        agent.train([Fast(), PlotState()])

    def test_train_plotsteps(self):
        agent = easyagents.PpoAgent("CartPole-v0")
        agent.train([Fast(), PlotSteps()])

    def test_train_multiple_subplots(self):
        agent = easyagents.PpoAgent("CartPole-v0")
        agent.train([Fast(), PlotState(), PlotRewards(), PlotLoss(), PlotSteps()])

"""
    def test(self):
        ppoAgent = easyagents.PpoAgent(gym_env_name='CartPole-v0', fc_layers=(500, 500, 500))
        ppoAgent.train([PlotLoss(ylim=(0.01, 100)), PlotSteps(), PlotRewards()],
                       learning_rate=0.0001,
                       num_iterations=2500, num_epochs_per_iteration=5, max_steps_per_episode=50,
                       num_iterations_between_eval=10, num_episodes_per_eval=10)
"""

if __name__ == '__main__':
    unittest.main()
