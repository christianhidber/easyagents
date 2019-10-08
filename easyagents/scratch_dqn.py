# Copyright 2018 Tensorforce Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os

import tensorflow as tf

from tensorforce.agents import Agent
from tensorforce.environments import Environment
from tensorforce.execution import Runner


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(v=tf.logging.ERROR)


def _create_network_specification(fc_layers):
    """Creates a tensorforce network specification based on the layer specification given in self.model_config"""
    result= []
    layer_sizes = fc_layers
    for layer_size in layer_sizes:
        result.append(dict(type='dense', size=layer_size, activation='relu'))
    return result

def main():
    # Create an OpenAI-Gym environment
    environment = Environment.create(environment='gym', level='CartPole-v1')
    network = _create_network_specification((100,))

    # Create a PPO agent
    agent = Agent.create(agent='dueling_dqn', environment=environment,network=network)
    runner = Runner(agent=agent, environment=environment)
    runner.run(num_episodes=10000)
    runner.close()


if __name__ == '__main__':
    main()
