"""Implements a creature that would exist in the NEAT algorithm."""
import numpy as np

from neat.genome import Genome, ConnectionGene, NodeGene, Phenotype
from neat.graph import Sensor, Output

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

    def __init__(self, n_inputs=None, n_outputs=None):
        """Create a creature.

        If both n_inputs and n_outputs are set, the creature initially has a fully
        connected neural network with n_inputs input nodes and n_outputs
        output nodes.
        Typically you want to set these parameters when you are first creating
        the seed creature for a population. If you want to create a copy be
        sure to use the copy() method.

        Arguments:
            n_inputs: How many inputs to expect.
            n_outputs: How many outputs are expected - also how many actions the creature can take.
        """
        # This option is here to support copying.
        if n_inputs is None or n_outputs is None:
            self.genotype = None
            self.phenotype = None

            return

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

        self.genotype = genome
        self.phenotype = Phenotype(genome)
        self.fitness = 0
        self.species = None

    def copy(self):
        """Make a copy of a creature.

        Returns: the copy of the creature.
        """
        copy = Creature()
        copy.genotype = self.genotype.copy()
        copy.phenotype = self.phenotype.copy()

        return copy

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
        max_genome_length = max(len(self.genotype), len(other_creature.genotype))
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
            mean_difference += abs(gene1.connection.weight - gene2.connection.weight)

        return mean_difference / len(aligned_genes)

    def align_genes(self, other_creature):
        """Find the aligned, disjoint, and excess genes of two creatures.

        Arguments:
            other_creature: the other creature to align genotypes with.

        Returns: a 3-tuple where the elements are a list of aligned genes,
                 disjoint genes, and excess genes. The aligned genes element
                 itself is also a tuple which contains the pairs of aligned
                 genes.
        """
        return self.genotype.align_genes(other_creature.genotype)

    def __cmp__(self, other_creature):
        if self.fitness < other_creature.fitness:
            return -1
        elif self.fitness > other_creature.fitness:
            return 1
        else:
            return 0

    def __lt__(self, other_creature):
        return self.__cmp__(other_creature) < 0
