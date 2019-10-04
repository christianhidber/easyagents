## Release notes
* v1.1 [19Q4]
    * 1.1.19: 
        * jupyter plotting performance improved
        * plot.ToMovie with support for animated gifs 
    * 1.1.18: tensorforce backend (ppo, reinforce)
    * 1.1.11:
        * plot.StepRewards, plot.Actions
        * default_plots parameter (instead of default_callbacks)
        
* v1.0.1 [19Q3]
    * api based on pluggable backends and callbacks (for plotting, logging, training durations)
    * backend: tf-agents, default
    * algorithms: dqn, ppo, random
    * plots: State, Loss (including actor-/critic loss), Steps, Rewards
    * support for creating a mp4 movie (plot.ToMovie) 
* v0.1 [19Q2]
    * prototype implementation / proof of concept
    * hard-wired support for Ppo, Reinforce, Dqn on tf-agents
    * hard-wired plots for loss, sum-of-rewards, steps and state rendering 
    * hard-wired mp4 rendering
    
## Design guidelines
* separate "public api" from concrete implementation using a frontend / backend architecture 
  (inspired by scikit learn, matplotlib, keras)
* pluggable backends
* extensible through callbacks (inspired by keras). separate callback types for training, evaluation and monitoring
* pre-configurable, algorithm specific train & play loops 
    
## Class diagram
![ClassDiagram](ClassDiagram.png)
    
    
## Glossary
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
 