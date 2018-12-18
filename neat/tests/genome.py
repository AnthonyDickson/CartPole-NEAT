"""Unit tests for the genome module."""

import random
import unittest

from neat.genome import Genome, NodeGene, ConnectionGene, Phenotype
from neat.graph import Sensor, Output, Hidden

class GenomeUnitTest(unittest.TestCase):
    """Test cases for the genome module unit test suite."""

    @staticmethod
    def generate_genome():
        """Generate a test genome.

        Returns a genotype similar to one found in the origin NEAT paper.
        """
        genome = Genome()

        nodes = [NodeGene(Sensor()) for _ in range(3)]
        nodes.append(NodeGene(Output()))
        nodes.append(NodeGene(Hidden()))

        connections = []

        for input_node_id in [0, 2, 4]:
            connections.append(ConnectionGene(3, input_node_id))

        for input_node_id in [0, 1, 3]:
            connections.append(ConnectionGene(4, input_node_id))

        genome.add_genes(nodes)
        genome.add_genes(connections)

        return genome

    def test_genome_innovation_number_generation(self):
        """Test if genes are assigned the correct innovation numbers."""
        genome = GenomeUnitTest.generate_genome()

        # Connection genes are all unique so their innovation numbers should just increase
        # monotonically starting from one.
        for i, gene in enumerate(genome.connection_genes):
            self.assertEqual(i, gene.innovation_number, \
                'Expected an innovation number of %d, but got %d.' % (i, gene.innovation_number))

    def test_genome_to_phenotype(self):
        """Test if genomes are correctly converted to their corresponding phenotype."""
        genome = GenomeUnitTest.generate_genome()

        phenotype = Phenotype(genome)

        x = [1, 1, 1]
        self.assertGreaterEqual(phenotype.compute(x), 0)

    def test_genome_copy(self):
        """Test if genomes copy and perform as expected."""
        genome = GenomeUnitTest.generate_genome()

        phenotype1 = Phenotype(genome)
        phenotype2 = Phenotype(genome.copy())

        x = [1, 1, 1]
        self.assertEqual(phenotype1.compute(x), phenotype2.compute(x))

if __name__ == '__main__':
    random.seed(42)

    unittest.main()
