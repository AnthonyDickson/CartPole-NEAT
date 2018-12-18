"""Implements the NEAT algorithm."""

from time import time

import numpy as np

from neat.creature import Creature
from neat.species import Species

class NeatAlgorithm:
    """An implementation of the NEAT algorithm based off the original paper."""

    def __init__(self, env, n_pops=150):
        self.env = env
        self.n_pops = n_pops
        self.population = self.init_population(env.observation_space.shape[0], env.action_space.n)
        self.species = set()

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

        step_msg_format = \
            "{:03d}/{:03d} - steps: {:02d} - step time {:02.4f}s"

        episode_complete_msg_format = "{:03d}/{:03d} - avg. steps: {:.2f} "\
            "- avg. step time: {:02.4f}s - avg. fitness: {:.4f} - total time: {:02.2f}s"

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
                        creature.fitness = step + 1
                        step_history.append(step + 1)
                        print(step_msg_format.format(\
                            pop_i + 1, self.n_pops, step + 1, time() - pop_start), end='\r')

                        break
                else:
                    creature.fitness = n_steps

            self.process_episode()

            avg_fitness = sum(c.fitness for c in self.population) / self.n_pops

            avg_steps = np.mean(step_history)
            total_episode_time = time() - episode_start
            avg_step_time = total_episode_time / self.n_pops

            print(episode_complete_msg_format.format(self.n_pops, \
                    self.n_pops, avg_steps, avg_step_time, total_episode_time, avg_fitness))

        print('Total run time: {:.2f}s - avg. steps: {:.2f} - best steps: {}'.format(\
            time() - sim_start, np.mean(step_history), np.max(step_history)))
        print()

    def process_episode(self):
        """Do the post-episode stuff such as speciating, adjusting creature fitness,
        crossover etc.
        """
        map(self.speciate, self.population)
        map(self.adjust_fitness, self.population)
        self.crossover(self.population)
        map(self.mutate, self.population)

    def speciate(self, creature):
        """Place a creature into a species, or create a new species if no
        suitable species exists.

        Arguments:
            creature: the creature to place into a species.
        """
        for species_i in self.species:
            if creature.distance(species_i.representative) < Species.compatibility_threshold:
                species_i.add(creature)
                creature.species = species_i

                break
        else:
            new_species = Species()
            new_species.members.add(creature)
            new_species.representative = creature
            self.species.add(new_species)
            creature.species = new_species

    def adjust_fitness(self, creature):
        """Update the creature's fitness with the adjusted (shared) fitness.

        Arguments:
            creature: the creature whose fitness should be adjusted. The
                      creature must be already speciated.
        """
        adjusted_fitness = creature.fitness / len(creature.species)
        creature.fitness = adjusted_fitness

    def crossover(self, speciated_population):
        """Perform crossover on the population.

        Arguments:
            speciated_population: the population after it has been partitioned
            into species.

        Returns: the new population.
        """
        pass

    def mutate(self, creature):
        """Mutate the given creature's genotype.

        Arguments:
            creature: the creature to be mutated.
        """
        pass
