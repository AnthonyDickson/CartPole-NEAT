"""Implements the NEAT algorithm."""

from time import time

import numpy as np

from neat.creature import Creature

class NeatAlgorithm:
    """An implementation of the NEAT algorithm based off the original paper."""

    def __init__(self, env, n_pops=150):
        self.env = env
        self.n_pops = n_pops
        self.population = self.init_population(env.observation_space.shape[0], env.action_space.n)

    def init_population(self, n_inputs, n_outputs):
        """Create a population of n individuals.

        Each individual is initially has a fully connected neural network with
        n_inputs neurons in the input layer and n_outputs neurons in the
        output layer.

        Arguments:
            n_inputs: How many inputs to expect. An observation in CartPole
                      has four dimensions, so in this case n_inputs would be four.
            n_outputs: How many outputs to expect. CartPole has two actions in
                      its action space, so in this case n_inputs would be two.

        Returns: the intialised population that is ready for use.
        """
        creature = Creature(n_inputs, n_outputs)
        population = [creature.copy() for _ in range(self.n_pops)]

        return population

    def train(self, n_episodes=100, n_steps=200):
        """Train species of individuals.

        Arguments:
            n_episodes: The number of episodes to trian for.
            n_steps: The maximum number of steps per individual per episode.
        """
        sim_start = time()

        for episode in range(n_episodes):
            step_history = []
            episode_start = time()

            print('Episode {:02d}/{:02d}'.format(episode + 1, n_episodes))

            for pop_i, creature in enumerate(self.population):
                observation = self.env.reset()
                pop_start = time()

                for step in range(n_steps):
                    action = creature.get_action(observation)
                    observation, reward, done, info = self.env.step(action)

                    if done:
                        step_history.append(step)
                        print("{:03d}/{:03d} - steps: {:02d} - step time {:02.4f}s".format(\
                            pop_i + 1, self.n_pops, step + 1, time() - pop_start), end='\r')

                        break

            avg_steps = np.mean(step_history)
            total_episode_time = time() - episode_start
            avg_step_time = total_episode_time / self.n_pops
            print("{:03d}/{:03d} - avg. steps: {:.2f} - avg. step time: {:02.4f}s - "\
                "total time: {:02.2f}s".format(self.n_pops, self.n_pops, avg_steps, avg_step_time,\
                total_episode_time))

        print('Total run time: {:.2f}s - avg. steps: {:.2f} - best steps: {}'.format(\
            time() - sim_start, np.mean(step_history), np.max(step_history)))
        print()
