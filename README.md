### Reinforcement Learning for Practitioners (v1)
![Travis_Status](https://travis-ci.com/christianhidber/easyagents.svg?branch=v1)

Status: under active development, breaking changes may occur

![EasyAgents logo](images/EazyAgentsIcon.png)

EasyAgents is a high level reinforcement learning api, written in Python and running on top of
[OpenAI gym](https://github.com/openai/gym) using algorithms implemented in 
[tf-Agents](https://github.com/tensorflow/agents), [OpenAI baselines](https://github.com/openai/baselines)
and [huskarl](https://github.com/danaugrs/huskarl).

### Use EasyAgents if
* you have implemented your own environment and want to experiment with it
* you want try out different libraries and algorithms, but don't want to learn
  the details of each implementation
* you are looking for an easy and simple way to get started with reinforcement learning  

Try it on colab:
* [Cartpole on colab](https://colab.research.google.com/github/christianhidber/easyagents/blob/v1/jupyter_notebooks/easyagents_cartpole.ipynb)
  introduction: training, plotting & switching algorithms. based on the classic reinforcement learning example 
   balancing a stick on a cart.
* [Berater on colab](https://colab.research.google.com/github/christianhidber/easyagents/blob/v1/jupyter_notebooks/easyagents_berater.ipynb)
  example of a custom environment & training. gym environment based on a routing problem.
* [LineWorld on colab](https://colab.research.google.com/github/christianhidber/easyagents/blob/v1/jupyter_notebooks/easyagents_line.ipynb)
  implement your own environment, workshop example [work in progress]

In collaboration with [Oliver Zeigermann](http://zeigermann.eu/). 

### Scenario: simple (quick test, plot state)
````
from easyagents.agents import PpoAgent
from easyagents.callbacks import plot, duration

ppoAgent = PpoAgent('CartPole-v0')
ppoAgent.train([plot.State(), duration.Fast()])
````
![Scenario_Simple](images/Scenario_simple.png)

### Scenario: more detailed (custom training, network, movie)
````
from easyagents.agents import PpoAgent
from easyagents.callbacks import plot, duration

ppoAgent = PpoAgent( 'Orso-v1', fc_layers=(500,500,500))

ppoAgent.train(learning_rate=0.0001,
               [plot.State(), plot.Rewards(), plot.Loss(), plot.Steps(), plot.ToMovie()], 
               num_iterations = 500, max_steps_per_episode = 50,
               default_callbacks=False )
````

[![Scenario_Detailed](images/Scenario_detailed.png)](https://raw.githubusercontent.com/christianhidber/easyagents/v1-chh/images/Scenario_detailed.mp4)


## Guiding Principles
* **easily train, evaluate & debug policies for (you own) gym environment** over "designing new algorithms"
* **simple & consistent** over "flexible & powerful"
* **inspired by keras**: 
    * same api across all algorithms
    * support different implementations of the same algorithm
    
#### Design ideas
* separate "public api" from concrete implementation using a frontend / backend architecture 
  (inspired by scikit learn, matplotlib, keras)
* pluggable backends
* extensible through callbacks (inspired by keras). separate callback types for training, evaluation and monitoring
* pre-configurable, algorithm specific train & play loops 

## Installation
Install from pypi using pip:

```python
pip install easyagents-v1
```

## Vocabulary
[some terms explained](vocabulary.md)

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
