"""Unit tests for the genome module."""

import random
import unittest

from neat.genome import Genome, NodeGene, ConnectionGene, Phenotype
from neat.graph import Sensor, Output, Hidden, Connection

class GenomeUnitTest(unittest.TestCase):
    """Test cases for the genome module unit test suite."""

    def test_genome_innovation_number_generation(self):
        """Test if genes are assigned the correct innovation numbers."""
        genome = Genome()

        nodes = [NodeGene(Sensor) for _ in range(3)]
        nodes.append(NodeGene(Output))
        nodes.append(NodeGene(Hidden))

        connections = []

        for input_node_id in [0, 2, 4]:
            connections.append(ConnectionGene(input_node_id, 3))

        for input_node_id in [0, 1, 3]:
            connections.append(ConnectionGene(input_node_id, 4))

        genome.add_genes(nodes)
        genome.add_genes(connections)

        # Connection genes are all unique so their innovation numbers should just increase 
        # monotonically starting from one.
        for i, gene in enumerate(genome.connections):
            self.assertEqual(i, gene.innovation_number, \
                'Expected an innovation number of %d, but got %d.' % (i, gene.innovation_number))

    def test_genome_to_phenotype(self):
        """Test if genomes are correctly converted to their corresponding phenotype."""
        genome = Genome()

        nodes = [NodeGene(Sensor) for _ in range(3)]
        nodes.append(NodeGene(Output))
        nodes.append(NodeGene(Hidden))

        connections = []

        for input_node_id in [0, 2, 4]:
            connections.append(ConnectionGene(input_node_id, 3))

        for input_node_id in [0, 1, 3]:
            connections.append(ConnectionGene(input_node_id, 4))

        genome.add_genes(nodes)
        genome.add_genes(connections)

        phenotype = Phenotype(genome)

        x = [1, 1, 1]
        self.assertGreaterEqual(phenotype.compute(x), 0)

    def test_genome_copy(self):
        """Test if genomes copy and perform as expected."""
        genome = Genome()

        nodes = [NodeGene(Sensor) for _ in range(3)]
        nodes.append(NodeGene(Output))
        nodes.append(NodeGene(Hidden))

        connections = []

        for input_node_id in [0, 2, 4]:
            connections.append(ConnectionGene(input_node_id, 3))

        for input_node_id in [0, 1, 3]:
            connections.append(ConnectionGene(input_node_id, 4))

        genome.add_genes(nodes)
        genome.add_genes(connections)

        phenotype1 = Phenotype(genome)
        phenotype2 = Phenotype(genome.copy())

        x = [1, 1, 1]
        self.assertEqual(phenotype1.compute(x), phenotype2.compute(x))

if __name__ == '__main__':
    random.seed(42)

    unittest.main()
    