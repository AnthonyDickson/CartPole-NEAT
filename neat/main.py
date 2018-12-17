"""Implements the NEAT algorithm."""

from functools import reduce
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
                        creature.fitness = step + 1
                        step_history.append(step + 1)
                        print("{:03d}/{:03d} - steps: {:02d} - step time {:02.4f}s".format(\
                            pop_i + 1, self.n_pops, step + 1, time() - pop_start), end='\r')

                        break
                else:
                    creature.fitness = n_steps

            species = self.speciate()

            for creature in self.population:
                self.fitness(creature, species)

            avg_fitness = sum(c.fitness for c in self.population) / self.n_pops

            avg_steps = np.mean(step_history)
            total_episode_time = time() - episode_start
            avg_step_time = total_episode_time / self.n_pops
            print("{:03d}/{:03d} - avg. steps: {:.2f} - avg. step time: {:02.4f}s - "\
                "avg. fitness: {:.4f} - total time: {:02.2f}s".format(self.n_pops, \
                    self.n_pops, avg_steps, avg_step_time, total_episode_time, avg_fitness))

        print('Total run time: {:.2f}s - avg. steps: {:.2f} - best steps: {}'.format(\
            time() - sim_start, np.mean(step_history), np.max(step_history)))
        print()

    def distance(self, creature, other_creature):
        """Calculate the distance (or difference) between the genes of two
        different creatures.

        Arguments:
            creature: the first creature to compare.
            other_creature: the other creature to compare.

        Returns: the distance between the two creatures' genes.
        """ 
        c1 = 1.0
        c2 = 1.0
        c3 = 1.0
        N = max(len(creature.genotype), len(other_creature.genotype))
        aligned, disjoint, excess = self.gene_alignment(creature, other_creature)
        
        avg_w_diff = 0

        for conn1, conn2 in aligned:
            avg_w_diff += conn1.connection.weight - conn2.connection.weight
        
        avg_w_diff /= len(aligned)

        return c1 * len(disjoint) / N + c2 * len(excess) / N + c3 * abs(avg_w_diff)

    def gene_alignment(self, creature, other_creature):
        """Find the aligned, disjoint, and excess genes of two creatures.

        Arguments:
            creature: the first creature to compare.
            other_creature: the other creature to compare.

        Returns: a 3-tuple where the elements are a list of aligned genes, 
                 disjoint genes, and excess genes.
        """
        conn1 = creature.genotype.connection_genes
        conn2 = other_creature.genotype.connection_genes

        n = min(len(conn1), len(conn2))
        N = max(len(conn1), len(conn2))

        creature_max_innovation = max(conn1, key=lambda c: c.innovation_number).innovation_number
        other_max_innovation = max(conn2, key=lambda c: c.innovation_number).innovation_number
        threshold = min(creature_max_innovation, other_max_innovation)
        aligned_genes = []
        disjoint_genes = []
        excess_genes = []

        for i in range(n):
            if conn1[i].innovation_number != conn2[i].innovation_number:
                if conn1[i].innovation_number <= threshold:
                    disjoint_genes.append(conn1[i])  
                else:
                    excess_genes.append(conn1[i])
                if conn2[i].innovation_number <= threshold: 
                    disjoint_genes.append(conn2[i])
                else:
                    excess_genes.append(conn2[i]) 
            else:
                aligned_genes.append((conn1[i], conn2[i]))

        return aligned_genes, disjoint_genes, excess_genes

    def speciate(self):
        """Partition population into species.
        
        Returns: a dictionary where the keys are the species and the values 
                 are creatures in a given species.
        """
        species = {}

        for creature in self.population:
            for s in species:
                if self.distance(creature, species[s][0]) < 10:
                    species[s].append(creature)
                    creature.species = s

                    break
            else:
                creature.species = len(species)
                species[len(species)] = [creature]

        return species

    def fitness(self, speciated_creature, species, in_place=True):
        """Calculate the adjusted fitness score for a creature.

        Arguments:
            speciated_creature: the creature that has been assigned a species.
            species: the dictionary of species.
            in_place: whether to assign the fitness score directly or to return it.

        Returns: the fitness score for the given creature if in_place is True,
                 otherwise None.
        """
        adjusted_fitness = speciated_creature.fitness / len(species[speciated_creature.species])

        if in_place:
            speciated_creature.fitness = adjusted_fitness
        else:
            return adjusted_fitness

    def crossover(self, speciated_population):
        """Perform crossover on the population.

        Arguments:
            speciated_population: the population after it has been partitioned 
            into species.
        
        Returns: the new population.
        """
        pass

    def mutate(self, creature, in_place=True):
        """Mutate the given creature's genotype.

        Arguments:
            creature: the creature to be mutated.
            in_place: whether to modify the creature directly or to modify a 
                      copy of the creature.

        Returns: None if in_place is True, otherwise returns the mutated copy.
        """
        pass