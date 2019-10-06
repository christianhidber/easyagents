import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.cem import CEMAgent
from rl.memory import EpisodeParameterMemory

ENV_NAME = 'CartPole-v0'

import rl.callbacks


class MyCallback(rl.callbacks.Callback):

    def on_batch_begin(self, batch, logs=None):
        print('on_batch_begin')

    def on_epoch_begin(self, epoch, logs=None):
        print('on_batch_begin')

    def on_train_batch_begin(self, batch, logs=None):
        print('on_train_batch_begin')

    def on_train_begin(self, logs=None):
        print('on_train_begin')

    def on_test_batch_begin(self, batch, logs=None):
        print('on_test_batch_begin')

    def on_predict_batch_begin(self, batch, logs=None):
        print('on_predict_batch_begin')

    def on_predict_begin(self, logs=None):
        print('on_predict_begin')

    def on_action_begin(self, action, logs=None):
        if logs is None:
            logs = {}
        #print('on_action_begin')

    def on_episode_begin(self, episode, logs=None):
        if logs is None:
            logs = {}
        #print('on_episode_begin')

    def on_step_begin(self, step, logs=None):
        if logs is None:
            logs = {}
        #print('on_step_begin')


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)

nb_actions = env.action_space.n
obs_dim = env.observation_space.shape[0]

# Option 1 : Simple model
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(nb_actions))
model.add(Activation('softmax'))

# Option 2: deep network
# model = Sequential()
# model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
# model.add(Dense(16))
# model.add(Activation('relu'))
# model.add(Dense(16))
# model.add(Activation('relu'))
# model.add(Dense(16))
# model.add(Activation('relu'))
# model.add(Dense(nb_actions))
# model.add(Activation('softmax'))


print(model.summary())


# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = EpisodeParameterMemory(limit=1000, window_length=1)

cem = CEMAgent(model=model, nb_actions=nb_actions, memory=memory,
               batch_size=50, nb_steps_warmup=2000, train_interval=50, elite_frac=0.05)
cem.compile()

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
mycallback = MyCallback()
for i in range(0,100):
    print(f'iteration={i}')
    cem.fit(env, nb_steps=10000, visualize=False, verbose=0)
    cem.test(env, nb_episodes=3, visualize=False)
