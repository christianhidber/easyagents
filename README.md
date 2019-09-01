### Reinforcement Learning for Practitioners (v1-alpha)
![Travis_Status](https://travis-ci.com/christianhidber/easyagents.svg?branch=v1)

Status: under active development, breaking changes may occur

![EasyAgents logo](images/EazyAgentsIcon.png)

EasyAgents is a high level reinforcement learning api, written in Python and running on top of
[OpenAI gym](https://github.com/openai/gym) using algorithms implemented in 
[tf-Agents](https://github.com/tensorflow/agents) and [OpenAI baselines](https://github.com/openai/baselines).

### Use EasyAgents if
* you are looking for an easy and simple way to get started with reinforcement learning
* you have implemented your own environment and want to experiment with it
* you want mix and match different implementations and algorithms

Try it on colab:
* [Cartpole on colab](https://colab.research.google.com/github/christianhidber/easyagents/blob/v1/jupyter_notebooks/easyagents_cartpole.ipynb)
  (introduction. the classic reinforcement learning example balancing a stick on a cart)
* [Berater on colab](https://colab.research.google.com/github/christianhidber/easyagents/blob/v1/jupyter_notebooks/easyagents_berater.ipynb)
  (example of a custom environment & training. gym environment based on a routing problem)
* [LineWorld on colab](https://colab.research.google.com/github/christianhidber/easyagents/blob/v1/jupyter_notebooks/easyagents_line.ipynb)
  (implement your own environment, workshop example)

In collaboration with [Oliver Zeigermann](http://zeigermann.eu/). 


## Ideas for v1

### Guiding Principles
* **easily train, evaluate & debug policies for (you own) gym environment** over "designing new algorithms"
* **simple & consistent** over "flexible & powerful"
* **inspired by keras**: 
    * same api across all algorithms
    * support different implementations of the same algorithm

### Scenarios
* Simple
````
agent = PpoAgent( "LineWorld-v0" )
agent.train( SingleEpisode() )
agent.train()
agent.save(...)
agent.load(...)
agent.play()
````
* Advanced
````
agent = PpoAgent( "LineWorld-v0", fc_layers=(500,250,50) )
agent.train( train=[Fast(), ModelCheckPoint(), ReduceLROnPlateau(), TensorBoard()],
             play=[JupyterStatistics(), JupyterRender(), Mp4()],
             api=[AgentApi()] )
````
    
### Design ideas
* separate "public api" from concrete implementation using a frontend / backend architecture 
  (inspired by scikit learn, matplotlib, keras)
* pluggable backends
* extensible through callbacks (inspired by keras). separate callback types for training, evaluation and monitoring
* pre-configurable, algorithm specific train & play loops 

### Installation
Install from pypi using pip:

```python
pip install easyagents-v1
```

## Vocabulary

Here's a list of terms in the reinforcement learning space, explained in a colloquial way. The explanations are typically inprecise and just try to convey the general idea (if you find they are wrong or a term is missing: please let me know,
moreover the list only contains terms that are actually used for this project)

| term                          | explanation                           |
| ---                           | ---                                   |
| action                        | A game command to be sent to the environment. Depending on the game engine actions can be discrete (like left/reight/up/down buttons or continuous like 'move 11.2 degrees to the right')|
| batch                         | a subset of the training examples. Typically the training examples are split into batches of equal size.  |
| episode                       | 1 game played. A sequence of (state,action,reward) from an initial game state until the game ends.        |
| environment (aka game engine) | The game engine, containing the business logic for your problem. RL algorithms create an instance of the environment and play against it to learn a policy. |
| epoch                         | 1 full training step over all examples. A forward pass followed by a backpropagation for all training examples (batchs). |
| iterations                    | The number of passes needed to process all batches (=#training_examples/batch_size)                       |
| observation (aka game state)  | All information needed to represent the current state of the environment.                                 |
| optimal policy                | A policy that 'always' reaches the maximum number of points. Finding good policies for a game about which we know (almost) nothing else is the goal of reinforcement learning. Real-life algorithms typically don't find an optimal policy, striving for a local optimum.           |
| policy (aka gaming strategy)  | The 'stuff' we want to learn. A policy maps the current game state to an action. Policies can be very dump like one that randomly chooses an arbitrary action, independent of the current game state. Or they can be clever, like an that maximizes the reward over the whole game.      |
| training example              | a state together with the desired output of the neural network. For an actor network thats (state, action), for a value network (state, value). |

## Don't use EasyAgents if

* you would like to leverage implementation specific advantages of an algorithm
* you want to do distributed or in parallel reinforcement learning

## Note

* This repository is under active development and in an early stage. 
  Thus any- and everything may (and probably should) change.
* If you have any difficulties in installing or using easyagents please let us know. 
  We'll try to do our best to help you.
* Any ideas, help, suggestions, comments etc in python / open source development / reinforcement learning / whatever
  are more than welcome. Thanks a lot in advance.
