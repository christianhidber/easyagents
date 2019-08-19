import numpy as np
import math

import gym
from gym import error, spaces, utils

import matplotlib.pyplot as plt
from IPython.display import display, clear_output

import easyagents
from easyagents.easyenv import register_with_gym
from easyagents.tfagents import PpoAgent
from easyagents.config import Training

LEFT = 0
RIGHT = 1


class LineWorldEnv(gym.Env):

    # constructor sets up the properties of the environment
    # most important action space, observation space and reward range
    def __init__(self, state=[10, 0, 0, 5, 0, 2, 15]):
        self.state = np.array(state)
        # the agent can perform  different actions
        number_of_actions = 2
        self.action_space = spaces.Discrete(number_of_actions)

        self.size_of_world = len(state)

        ### OZ->CHH: Hier würden wir was hinschreiben, damit es zur Observation passt, das müssten die Teilnehmer dann ebenfalls umbauen
        # Für observation gleiche Werte für alles

        # Observation size = 5
        # Min = 0
        # Max = 10

        # Constante rein in v0

        # Ableiten von State in get observation

        # Plot arbeitet auf State, nicht observation

        # the environment's state is described by the position of the agent and the remaining rewards
        low = np.append([0], np.full(self.size_of_world, 0))
        high = np.append([self.size_of_world - 1], np.full(self.size_of_world, 15))
        self.observation_space = spaces.Box(low=low,
                                            high=high,
                                            dtype=np.float32)

        self.reward_range = (-1, 1)
        # 32 is only theoretical, because we need to travel at least 9 steps, leaving us with 23 practically
        self.optimum = self.state.sum() - 9

        self._figure = None

        self.reset()

    # OZ->CHH: Für die Übung würde hier nur ein flaches np.array herauskommen mit Nullen drin, geht dann natürlich nicht
    def get_observation(self):
        return np.append([self.pos], self.remaining_rewards)

    def reset(self):
        self.total_reward = 0
        self.done = False
        self.pos = math.floor(len(self.state) / 2)
        self.steps = 0

        self.remaining_rewards = np.array(self.state, copy=True)
        return self.get_observation()

    def step(self, action):
        if action == LEFT and self.pos != 0:
            self.pos -= 1
        elif self.pos < self.size_of_world - 1:
            self.pos += 1

        reward = self.remaining_rewards[self.pos] - 1
        normalized_reward = reward / self.optimum
        self.total_reward += normalized_reward
        self.remaining_rewards[self.pos] = 0

        if self.pos == 0 or self.total_reward * self.optimum <= -20:
            self.done = True
        self.steps += 1

        observation = self.get_observation()
        info = None
        return observation, normalized_reward, self.done, info

    def _render_ansi(self):
        return 'position: {position}, remaining rewards: {rewards}, total reward so far: {total}, normalized total reward: {normalized_total}, steps so far: {steps}, game done: {done}'.format(
            position=self.pos,
            rewards=self.remaining_rewards,
            total=self.total_reward * self.optimum,
            normalized_total=self.total_reward,
            done=self.done,
            steps=self.steps)

    def _render_to_ansi(self):
        return 'position: {position}, remaining rewards: {rewards}, total reward so far: {total}, normalized total reward: {normalized_total}, steps so far: {steps}, game done: {done}'.format(
            position=self.pos,
            rewards=self.remaining_rewards,
            total=self.total_reward * self.optimum,
            normalized_total=self.total_reward,
            done=self.done,
            steps=self.steps)

    def _render_to_figure(self):
        """ Renders the current state as a graph with matplotlib """
        if self._figure is not None:
            plt.close(self._figure)
        self._figure, ax = plt.subplots(1, figsize=(8, 4))
        x = np.arange(0, self.size_of_world, 1, dtype=np.uint8)
        y = self.remaining_rewards
        plt.plot([self.pos, self.pos], [0, 2], 'r^-')
        ax.scatter(x, y, s=75)
        self._figure.canvas.draw()

    def _render_to_human(self):
        """ show render_to_figure in a jupyter cell.
            the result of each call is rendered in the same cell"""
        clear_output(wait=True)
        self._render_to_figure()
        plt.pause(0.01)

    def _render_to_rgb(self):
        """ convert the output of render_to_figure to a rgb_array """
        self._render_to_figure()
        self._figure.canvas.draw()
        buf = self._figure.canvas.tostring_rgb()
        num_cols, num_rows = self._figure.canvas.get_width_height()
        plt.close(self._figure)
        self._figure = None
        result = np.fromstring(buf, dtype=np.uint8).reshape(num_rows, num_cols, 3)
        return result

    def render(self, mode='ansi'):
        if mode == 'ansi':
            return self._render_to_ansi()
        elif mode == 'human':
            return self._render_to_human()
        elif mode == 'rgb_array':
            return self._render_to_rgb()
        else:
            super().render(mode=mode)


env_name = "LineWorld-v0"
register_with_gym(gym_env_name=env_name, entry_point=LineWorldEnv, max_episode_steps=100)

training = Training(num_iterations=10,
                    num_episodes_per_iteration=5,
                    max_steps_per_episode=50,
                    num_epochs_per_iteration=5)

ppoAgent = PpoAgent(gym_env_name=env_name,
                    fc_layers=(500, 500),
                    training=training,
                    learning_rate=1e-4
                    )
ppoAgent.train()
ppoAgent.render_episodes(num_episodes=1, mode='ansi')
_ = ppoAgent.plot_episodes(ylim=[None, (0, 1), (0, 50)])
easyagents.agents._is_jupyter_active = True
ppoAgent.render_episodes_to_jupyter(num_episodes=1, fps=1)
