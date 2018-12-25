"""Implements the PSO (Particle Swarm Optimisation) algorithm."""
import random
from time import time

import numpy as np


class Particle:
    """A particle that takes part in PSO."""
    momentum = 0.9
    acceleration_self = 0.9
    acceleration_global_best = 0.9

    def __init__(self, creature):
        """Create a particle.

        Arguments:
            creature: The creature that this particle represents.
        """
        self.creature = creature
        self.original_phenotype = creature.phenotype.copy()
        self.best_phenotype = creature.phenotype.copy()
        self.best_fitness = -1
        self.fitness = -1
        self.weight_velocity = [0 for _ in
                                range(len(creature.phenotype.connections))]
        self.bias_velocity = [0 for _ in range(len(creature.phenotype.nodes))]

    def get_action(self, x):
        return self.creature.get_action(x)

    def evaluate(self, env, n_steps):
        observation = env.reset()
        episode_reward = 0

        for step in range(n_steps):
            action = self.get_action(observation)
            observation, reward, done, _ = env.step(action)
            episode_reward += reward

            if done:
                break

        self.fitness = episode_reward

    def next_state(self, best_particle):
        """Update the particle's state to the next one."""
        if self.fitness > self.best_fitness:
            self.best_fitness = self.fitness
            self.best_phenotype = self.creature.phenotype.copy()

        i = 0
        # TODO: Come up with better names for these variables.
        a = random.random()
        b = random.random()

        # TODO: Refactor duplicate code.
        for node, node_best, node_global_best in \
                zip(self.creature.phenotype.nodes.values(),
                    self.best_phenotype.nodes.values(),
                    best_particle.creature.phenotype.nodes.values()):
            diff_self = node_best.bias - node.bias
            diff_best = node_global_best.bias - node.bias

            self.bias_velocity[i] = Particle.momentum * self.bias_velocity[i] + \
                                    Particle.acceleration_self * a * diff_self + \
                                    Particle.acceleration_global_best * b * diff_best

            node.bias += self.bias_velocity[i]

            i += 1

        i = 0

        for connection, connection_best, connection_global_best in \
                zip(self.creature.phenotype.connections,
                    self.best_phenotype.connections,
                    best_particle.creature.phenotype.connections):
            diff_self = connection_best.weight - connection.weight
            diff_best = connection_global_best.weight - connection.weight

            self.weight_velocity[i] = Particle.momentum * self.weight_velocity[i] + \
                                      Particle.acceleration_self * a * diff_self + \
                                      Particle.acceleration_global_best * b * diff_best

            connection.weight += self.weight_velocity[i]

            i += 1

    def apply_changes(self):
        """Apply the best set of weights and biases to the creature."""
        self._apply(self.best_phenotype)

    def restore(self):
        """Restore the creature to its initial state."""
        self._apply(self.original_phenotype)

    def _apply(self, phenotype):
        """Apply the weights and biases in the given phenotype to the
        creature's phenotype.

        Arguments:
            phenotype: The phenotype with the weights and bias that the
                       creature's phenotype weights and biases should be set
                       to.
        """
        for creature_node, target_node in \
                zip(self.creature.phenotype.nodes.values(),
                    phenotype.nodes.values()):
            creature_node.bias = target_node.bias

        for creature_connection, target_connection in \
                zip(self.creature.phenotype.connections,
                    phenotype.connections):
            creature_connection.weight = target_connection.weight

    def __lt__(self, other):
        return self.fitness < other.fitness


class PSO:
    """Performs PSO on a genotype and optimises weight and biases."""

    def __init__(self, env, population):
        self.env = env
        self.population = [Particle(creature) for creature in population]
        self.best_particle = self.population[0]

    def train(self, n_episodes=10, n_steps=200):
        """Perform particle optimisation.

        Note: Modifies creature genotype and phenotype directly.

        Arguments:
            n_episodes: How many iterations of PSO to perform.
            n_steps: The maximum number of steps to run the environment for
                     each episode and particle.

        """
        fitness_history = []
        start = time()

        for episode in range(n_episodes):
            fitness_history.append([])
            episode_start = time()

            for i, particle in enumerate(self.population):
                particle.evaluate(self.env, n_steps)
                fitness_history[episode].append(particle.fitness)

                if particle.fitness > self.best_particle.fitness:
                    self.best_particle = particle

                particle.next_state(self.best_particle)

                print("\rEpisode {:03d}/{:03d} - Step {:03d}/{:03d} - "
                      "mean fitness: {:.2f} - median fitness: {:.2f} - "
                      "mean time per particle: {:.4f}s - species time: {:.4f}s"
                      .format(episode + 1, n_episodes,
                              i + 1, len(self.population),
                              np.mean(fitness_history[episode]),
                              np.median(fitness_history[episode]),
                              (time() - episode_start) / (i + 1),
                              time() - start),
                      end='')

    def apply(self):
        for particle in self.population:
            particle.apply_changes()

    def restore(self):
        for particle in self.population:
            particle.restore()
