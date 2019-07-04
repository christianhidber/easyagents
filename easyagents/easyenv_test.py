import unittest
import gym
import logging
import easyagents.easyenv
import easyagents.tfagents
from easyagents.config import TrainingDurationFast
from easyagents.config import TrainingDurationSingleEpisode
from easyagents.config import LoggingVerbose

logging.basicConfig(level=logging.DEBUG)

class TestEasyEnv(unittest.TestCase):

    def test_register(self):
        logging.basicConfig(level=logging.DEBUG)
        name = easyagents.easyenv.register('CartPole-v0')

        assert name == "Easy_CartPole-v0"

        env = gym.make(name)
        assert env is not None
        return

    def test_register_twiceSameName_success(self):
        easyagents.easyenv.register('CartPole-v0')
        easyagents.easyenv.register('CartPole-v0')
        return

    def test_register_twiceDifferentNames_fail(self):
        easyagents.easyenv.register('CartPole-v0')
        self.assertRaises(Exception, easyagents.easyenv.register, 'Dummy-v0')
        return

    def test_LoggingVerbose(self):
        ppo_agent = easyagents.tfagents.PpoAgent( 'CartPole-v0', training_duration=TrainingDurationFast(), logging=LoggingVerbose() )
        ppo_agent.train()
        return    
    
    def test_play_episode(self):
        ppo_agent = easyagents.tfagents.PpoAgent( 'CartPole-v0', training_duration=TrainingDurationSingleEpisode(), logging=LoggingVerbose() )
        ppo_agent.train()

        TestEasyEnv._step_callback_call_count=0
        (reward, steps) = ppo_agent.play_episode(step_callback)
        assert reward > 0
        assert steps > 0
        assert TestEasyEnv._step_callback_call_count>0
        return   

def step_callback(gym_env,action,state,reward,done,info):
    TestEasyEnv._step_callback_call_count +=1
    return

if __name__ == '__main__':
    unittest.main()