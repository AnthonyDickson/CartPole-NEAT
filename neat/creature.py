"""Implements a creature that would exist in the NEAT algorithm."""
import random

import numpy as np

from neat.genome import Genome, ConnectionGene, NodeGene
from neat.graph import Sensor, Output
from neat.phenotype import Phenotype


class Creature:
    """A creature that would exist in the NEAT algorithm."""

    # More values for the following variables are documented in the original
    # NEAT paper.

    # The following three variables are weights that affect how important
    # each of the three properties affect the distance metric between
    # two creatures.
    disjointedness_importance = 1.0
    excessivity_importance = 1.0
    weight_unsameness_importance = 3.0

    # The probability that the next offspring will be created through mutation
    # only.
    p_mutate_only = 0.25

    # The probability that a new creature will be created through crossover
    # only.
    p_mate_only = 0.2

    def __init__(self, n_inputs=None, n_outputs=None):
        """Create a creature.

        If both n_inputs and n_outputs are set, the creature initially has a
        fully connected neural network with n_inputs input nodes and n_outputs
        output nodes.
        Typically you want to set these parameters when you are creating a
        creature, and the option to leave them as None is mainly for internal
        use.

        Arguments:
            n_inputs: How many inputs to expect.
            n_outputs: How many outputs are expected - also how many actions
                    the creature can take.
        """
        self.raw_fitness = 0
        self.fitness = 0
        self._species = None
        self.name_suffix = None
        self.past_species = []
        self.age = 0

        if n_inputs is None or n_outputs is None:
            self.genotype = None
            self.phenotype = None
        else:
            self.genotype = Creature.fully_connected_nn(n_inputs, n_outputs)
            self.phenotype = Phenotype(self.genotype)

    @staticmethod
    def fully_connected_nn(n_inputs, n_outputs):
        """Generate a genotype representing a fully connected neural network.

        Arguments:
            n_inputs: How many inputs the network should have.
            n_outputs: How many outputs the network should have.

        Returns: The generated genotype.
        """
        genome = Genome()

        sensors = [NodeGene(Sensor()) for _ in range(n_inputs)]
        outputs = [NodeGene(Output()) for _ in range(n_outputs)]

        connections = []

        for input_id in range(n_inputs):
            for output_id in range(n_inputs, n_inputs + n_outputs):
                connections.append(ConnectionGene(output_id, input_id))

        genome.add_genes(sensors)
        genome.add_genes(outputs)
        genome.add_genes(connections)

        return genome

    def copy(self):
        """Make a copy of a creature.

        Returns: the copy of the creature.
        """
        copy = Creature()
        copy.genotype = self.genotype.copy()
        copy.phenotype = Phenotype(copy.genotype)

        copy.age = self.age
        copy.name_suffix = self.name_suffix
        copy.past_species = self.past_species
        copy.species = self.species

        return copy

    @property
    def composite_fitness(self):
        """The combination of raw fitness and adjusted fitness."""
        return self.raw_fitness + self.fitness

    @property
    def name(self):
        species = 'Unknown' if not self.species else self.species.name
        suffix = ' %s' % self.name_suffix if self.name_suffix else ''

        return species + suffix

    @property
    def scientific_name(self):
        n_sensors = len(self.phenotype.sensors)
        n_outputs = len(self.phenotype.outputs)
        n_hidden = len(self.phenotype.nodes) - (n_outputs + n_sensors)
        n_connections = len(self.phenotype.connections)
        n_recurrent = len(self.phenotype.recurrent_connections)

        return 'S%dH%dO%dC%dR%d' % (n_sensors, n_hidden, n_outputs,
                                    n_connections, n_recurrent)

    @property
    def species(self):
        return self._species

    @species.setter
    def species(self, new_species):
        if self.species:
            self.past_species.append(self.species.name)

        self._species = new_species

    def get_action(self, x):
        """Get the creature's action for the given input.

        Arguments:
            x: the input, typically an observation of the agent's environment.

        Returns: an integer representing the action the creature will take.
        """
        return np.argmax(self.phenotype.compute(x))

    def distance(self, other_creature):
        """Calculate the distance (or difference) between the genes of two
        different creatures.

        Arguments:
            other_creature: the creature to compared with.

        Returns: the distance between the two creatures' genes.
        """
        max_genome_length = max(len(self.genotype),
                                len(other_creature.genotype))
        aligned, disjoint, excess = self.align_genes(other_creature)

        disjointedness = len(disjoint) / max_genome_length
        excessivity = len(excess) / max_genome_length
        weight_unsameness = Creature.mean_weight_difference(aligned)

        return Creature.disjointedness_importance * disjointedness + \
               Creature.excessivity_importance * excessivity + \
               Creature.weight_unsameness_importance * weight_unsameness

    @staticmethod
    def mean_weight_difference(aligned_genes):
        """Calculate the average difference between a set of aligned genes.

        Arguments:
            aligned_genes: the list of the aligned connection gene pairs.

        Returns: the average weight difference between the aligned genes.
        """
        mean_difference = 0

        for gene1, gene2 in aligned_genes:
            mean_difference += abs(gene1.connection.weight -
                                   gene2.connection.weight)

        return mean_difference / len(aligned_genes)

    def adjust_fitness(self):
        """Update the creature's fitness with the adjusted (shared) fitness."""
        self.raw_fitness = self.fitness
        self.fitness = self.fitness / len(self.species)

    def align_genes(self, other_creature):
        """Find the aligned, disjoint, and excess genes of two creatures.

        Arguments:
            other_creature: the other creature to align genotypes with.

        Returns: a 3-tuple where the elements are a list of aligned genes,
                 disjoint genes, and excess genes. The aligned genes element
                 itself is also a tuple which contains the sets of aligned
                 genes for each creature.
        """
        dominance = self.composite_fitness - other_creature.composite_fitness

        return Genome.align_genes(self.genotype.connection_genes,
                                  other_creature.genotype.connection_genes,
                                  dominance)

    def mate(self, other):
        """Mate, it's time to mate. Create a baby creature from two creatures.

        Arguments:
                other: the other creature to mate with, mate.

        Returns: a new creature.
        """
        creature = Creature()
        mate_only = random.random() < Creature.p_mate_only
        mutate_only = random.random() < Creature.p_mutate_only

        if mutate_only:
            creature.genotype = self.genotype.copy()
        else:
            dominance = self.composite_fitness - other.composite_fitness
            creature.genotype = self.genotype.crossover(other.genotype,
                                                        dominance)

        if not mate_only:
            creature.mutate()

        creature.phenotype = Phenotype(creature.genotype)

        return creature

    def mutate(self):
        """Mutate a creature's genotype."""
        self.genotype.mutate()

    def __lt__(self, other_creature):
        return self.composite_fitness < other_creature.composite_fitness

    def __str__(self):
        return '%s (%s)' % (self.name, self.scientific_name)
