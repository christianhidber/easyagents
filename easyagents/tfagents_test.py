import unittest
import tensorflow as tf
from easyagents.tfagents import PpoAgent
from easyagents.config import TrainingDurationFast
from easyagents.config import TrainingDurationSingleEpisode
from easyagents.config import LoggingVerbose

class TestTfAgents(unittest.TestCase):

    def setUp(self):
        self.gym_env_name='CartPole-v0'

    def test_ppo_create(self):
        ppo_agent = PpoAgent( self.gym_env_name )
        self.assertIsNotNone( ppo_agent, "failed to create a tfagents.Ppo instance for " + self.gym_env_name )
        return

    def test_ppo_train(self):
        ppo_agent = PpoAgent( self.gym_env_name, training_duration=TrainingDurationFast() )
        ppo_agent.train()
        return

    def test_ppo_str(self):
        ppo_agent = PpoAgent( self.gym_env_name, training_duration=TrainingDurationFast() )
        result = str(ppo_agent)
        print(result)
        return       

    def test_render_episodes_to_mp4(self):
        ppo_agent = PpoAgent( 'CartPole-v0', training_duration=TrainingDurationSingleEpisode(), logging=LoggingVerbose() )
        ppo_agent.train()

        ppo_agent.render_episodes_to_mp4()
        return

    def test_plot_no_ylim(self):
        ppo_agent = PpoAgent( 'CartPole-v0', training_duration=TrainingDurationSingleEpisode(), logging=LoggingVerbose() )
        ppo_agent.train()

        ppo_agent.plot_average_rewards()
        return

    def test_plot_with_ylim(self):
        ppo_agent = PpoAgent( 'CartPole-v0', training_duration=TrainingDurationSingleEpisode(), logging=LoggingVerbose() )
        ppo_agent.train()

        ppo_agent.plot_average_rewards(ylim=[10,20])
        return

    def test_plot_with_log_scale(self):
        ppo_agent = PpoAgent( 'CartPole-v0', training_duration=TrainingDurationSingleEpisode(), logging=LoggingVerbose() )
        ppo_agent.train()

        ppo_agent.plot_average_rewards(scale='log')
        return

    def test_plot_losses_with_ylim(self):
        ppo_agent = PpoAgent( 'CartPole-v0', training_duration=TrainingDurationSingleEpisode(), logging=LoggingVerbose() )
        ppo_agent.train()

        ppo_agent.plot_losses(ylim=[10,20])
        return

    def test_render_episodes(self):
        ppo_agent = PpoAgent( 'CartPole-v0', training_duration=TrainingDurationSingleEpisode(), logging=LoggingVerbose() )
        ppo_agent.train()

        ppo_agent.render_episodes()
        return


if __name__ == '__main__':
    unittest.main()
