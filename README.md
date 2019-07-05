# EasyAgents (work in progress)

An easy and simple way to get started with reinforcement learning and experimenting with your
own game engine / environment. EasyAgents is based on OpenAI gym and
uses algorithms implemented by tfAgents or OpenAI baselines.

Try it on colab:
* [Cartpole on colab](https://colab.research.google.com/github/christianhidber/easyagents/blob/master/jupyter_notebooks/easyagents_cartpole.ipynb)
  (the classic reinforcement learning example balancing a stick on a cart)
* [Berater on colab](https://colab.research.google.com/github/christianhidber/easyagents/blob/master/jupyter_notebooks/easyagents_berater.ipynb)
  (an example of a gym environment implementation based on a routing problem)

## Example

Here's an example of the full code needed to run the tfagents implementation of Ppo on the cartpole example.

Install from pypi using pip:

```python
pip install easyagents
```

### Simplest case (no configuration)

```python
from easyagents.tfagents import PpoAgent

ppo_agent = PpoAgent( gym_env_name='CartPole-v0')
ppo_agent.train()
```

Points of interest:

* If you prefer the baselines implementation change the import to 'from easyagents.baselines import Ppo'.
  That's all no other changes are necessary.
* If you would like to see plots of the average returns and losses during training (in a jupyter notebook):

    ```python
    ppo_agent.plot_average_rewards()
    ppo_agent.plot_losses()
    ```

* By default every api call during training is logged, as well as a summary of every game played.
  You can restrict / extend logging to topic areas like 'agent api' or 'environment api' calls.

### With Configuration (layers, training, learning rate, evaluation)

```python
from easyagents.tfagents import PpoAgent
from easyagents.config import TrainingDuration

training_duration = TrainingDuration(   num_iterations = 2000,
                                        num_episodes_per_iteration = 100,
                                        num_epochs_per_iteration = 5,
                                        num_iterations_between_eval = 10,
                                        num_eval_episodes = 10 )
ppo_agent = PpoAgent(   gym_env_name='CartPole-v0',
                        fc_layers=(500,500,500),
                        training_duration=training_duration )
ppo_agent.train()
```

Points of interest:

* The signature of PpoAgent() stays the same across all implementations.
  Thus you can still switch to the OpenAI baselines implementation simply by substituting the import statement.
* You can also use the preconfigured TrainingDurationFast()
* All 'agent api' and all 'gym api' calls are logged. You can easly turn them individually on or off using

```python
  PpoAgents( ..., logging=Logging( log_agent=true, log_gym_api=false ), ...)
```

* You may also use the preconfigure LoggingVerbose() or LoggingSilent()

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

## For whom this is for (and for whom not)

* If you have a general understanding of reinforcement learning and you would like to try different
algorithms and/or implemenations in an easy way: this is for you.
* If you would like to play around with your own reinforcement learning problem and getting a feeling
if some of the well-known algorithms may be helpful: this is for you.
* If you are a reinforcement learning EXPERT or if you would like to leverage implementation specific
advantages of an algorithm: this is NOT for you.

## Note

* This is a prototype / proof of concept. Thus any- and everything may (and probably should) change.
* If you have any difficulties in installing or using easyagents please let me know. I'll try to do my best to help you.
* I am a fairly / very inexperienced in python and open source development. Any ideas, help, suggestions, comments etc are more than welcome. Thanks a lot in advance.

## Stuff to work on

This is just a heap of ideas, issues and stuff that needs work / should be thought about (besides all the stuff that isn't mentioned):

* using gin for configuration ?
* logenv.register should allow multiple environments to be registered
* support for baselines
* support for multiple agents (not just ppo)
