# EasyAgents (prototype / proof of concept)

An easy and simple way to get started with reinforcement learning and experimenting with your
own game engine / environment. EasyAgents is based on OpenAI gym and
uses algorithms implemented by tfAgents or OpenAI baselines.

## Example (implementation in progress)

Here's an example of the full code needed using easyagents to run the tfagents implementation of Ppo on the cartpole example:

### Simplest case (no configuration)

```python
from easyagents.tfagents import Ppo

ppoAgent = Ppo( 'CartPole-v0' )
ppoAgent.train()
```

If you prefer the baselines implementation change the import to 'from easyagents.baselines import Ppo'.
That's all no other changes are necessary.

### With Configuration

```python
from easyagents.tfagents import Ppo

ppoAgent = Ppo( 'CartPole-v0', fc_layers=(100,200) )
ppoAgent.train( num_training_episodes=100,
                num_training_episodes_per_iteration=10,
                num_eval_episodes=10,
                learning_rate=0.99,
                reward_discount_gamma=1 )
```

The signature of train() stays the same across all implementations. Thus even with the additional
arguments you can still switch to the OpenAI baselines implementation simply by substituting the
import statement.

## For whom this is for (and for whom not)

* If you have a general understanding of reinforcement learning and you would like to try different
algorithms and/or implemenations in an easy way: this is for you.
* If you would like to play around with your own reinforcement learning problem and getting a feeling
if some of the well-known algorithms may be helpful: this is for you.
* If you are a reinforcement learning EXPERT or if you would like to leverage implementation specific
advantages of an algorithm: this is NOT for you.

## Vocabulary

Here's a list of terms in the reinforcement learning space, explained in a colloquial way. The explanations are typically inprecise and just try to convey the general idea (if you find they are wrong or a term is missing: please let me know,
moreover the list only contains terms that are actually used for this project)

| term                          | explanation                           |
| ---                           | ---                                   |
| episode                       | 1 game played. A sequence of (state,action,reward) from an initial game state until the game ends. |
| training example              | a state together with the desired output of the neural network. For an actor network thats (state, action), for a value network (state, value). |
| batch                         | a subset of the training examples. Typically the training examples are split into batches of equal size. |
| iterations                    | The number of passes needed to process all batches (=#training_examples/batch_size) |
| epoch                         | 1 full training step over all examples. A forward pass followed by a backpropagation for all training examples (batchs). |
| action                        | A game command to be sent to the environment. Depending on the game engine actions can be discrete (like left/reight/up/down buttons or continuous like 'move 11.2 degrees to the right')|
| observation (aka game state)  | All information needed to represent the current state of the environment. RL   |
|environment (aka game engine)  | The game engine, containing the business logic for your problem. RL algorithms create an instance of the environment and play against it to learn a policy. |
|policy (aka gaming strategy)   | The 'stuff' we want to learn. A policy maps the current game state to an action. Policies can be very dump like one that randomly chooses an arbitrary action, independent of the current game state. Or they can be clever, like an that maximizes the reward over the whole game.|
|optimal policy                 | A policy that 'always' reaches the maximum number of points. Finding good policies for a game about which we know (almost) nothing else is the goal of reinforcement learning. Real-life algorithms typically don't find an optimal policy, striving for a local optimum. |

## Note

* This is a prototype / proof of concept. Thus any- and everything may (and probably should) change.
* If you have any difficulties in installing or using eazy_agents please let me know. I'll try to do my best to help you.
* I am a fairly / very inexperienced in python and open source development. Any ideas, help, suggestions, comments etc are more than welcome. Thanks a lot in advance.

## Example for an alternative design approach based on a fluent interface (not implemented)

At the core of this approach lies the observation that we have a very large number of configuration / hyperparameters,
typically yielding huge argument lists (which always confuse me)

This approach is inspired by <https://en.wikipedia.org/wiki/Fluent_interface#Python>.

### Fluent api: Simplest case (no configuration)

```python
from easyagents import EazyAgent

ezAgent = EazyAgent( 'CartPole-v0' )
ezAgent.train()
```

### Fluent api: With Configuration

```python
from easyagents import EazyAgent

ezAgent = EazyAgent( 'CartPole-v0' )
            .SetTfAgent('Ppo')
            .GuessNetworkArchitecture()     # .SetNetworkArchitecture(fc_layers=(100,200))
            .GuessTrainingDuration()        # .SetTrainingDuration( num_episodes=100, num_episodes_per_iteration=10 )
            .SetRewardDiscount(0.9)
            .SetLearningRate(0.99)
ezAgent.train()
```

Points of interest:

* 'GuessNetworkArchitecture' tries to guess a possible network architecture based on action- & obeservation space
given by the chosen gym environment.
* 'SetNetworkArchitecture' lets the user choose the exact network architecture.
* Calling the 'Guess*' is optional, if the corresponding 'Set*' method is not called then they are called implicitely.
* Note that even calling 'SetTfAgent' is optional. If not used, then 'GuessAgent' is called which would try to choose
  an agent again based on the currently chosen environment.
