
from easyagents.tfagents import PpoAgent
from easyagents.config import TrainingFast
from easyagents.config import Logging

ppo_agent = PpoAgent(gym_env_name='CartPole-v0', training=TrainingFast(), logging=Logging(plots=True))
ppo_agent.train()

import gym

c = gym.make('CartPole-v0')
r=c.render(mode='rgb_array')

print(type(r))

input('press enter to continue...')



#%%



