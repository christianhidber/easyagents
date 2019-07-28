
from easyagents.tfagents import PpoAgent
from easyagents.config import TrainingFast
from easyagents.config import Logging

ppo_agent = PpoAgent(gym_env_name='CartPole-v0', training=TrainingFast(num_eval_episodes=3))
ppo_agent.train()

input('press enter to continue...')



#%%



