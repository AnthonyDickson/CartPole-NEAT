import random
import unittest

from neat.genome import Gene,  Genome
from neat.graph import Sensor, Output, Hidden, Connection

class GenomeUnitTest(unittest.TestCase):
    def test_genome_innovation_number_generation(self):
        genome = Genome()

        nodes = [Gene(Sensor()) for _ in range(3)]
        nodes.append(Gene(Output()))
        nodes.append(Gene(Hidden()))

        connections = []

        for other in [nodes[0], nodes[2], nodes[4]]:
            connections.append(Gene(Connection(nodes[3].allele.id, other.allele.id)))

        for other in [nodes[0], nodes[1], nodes[3]]:
            connections.append(Gene(Connection(nodes[4].allele.id, other.allele.id)))

        genome.add_genes(nodes)
        genome.add_genes(connections)

        # Genes are all unique so their innovation numbers should just increase monotonically 
        # starting from one.
        for i, gene in enumerate(genome.genes):
            self.assertEqual(i + 1, gene.innovation_number, 'Expected an innovation number of %d, but got %d.' % (i, gene.innovation_number))

    def test_genome_to_phenome(self):
        genome = Genome()
        genes = [Gene(Sensor()) for _ in range(3)]
        genes.append(Gene(Output()))
        genes.append(Gene(Hidden()))

        genome.add_genes(genes)

        for other in [genes[0], genes[2], genes[4]]:
            genome.add_gene(Gene(Connection(genes[3].allele.id, other.allele.id)))

        for other in [genes[0], genes[1], genes[3]]:
            genome.add_gene(Gene(Connection(genes[4].allele.id, other.allele.id)))

        g1 = genome.get_phenome()

        x = [1, 1, 1]
        self.assertGreaterEqual(g1.compute(x), 0)

    def test_genome_copy(self):
        genome = Genome()
        genes = [Gene(Sensor()) for _ in range(3)]
        genes.append(Gene(Output()))
        genes.append(Gene(Hidden()))

        genome.add_genes(genes)

        for other in [genes[0], genes[2], genes[4]]:
            genome.add_gene(Gene(Connection(genes[3].allele.id, other.allele.id)))

        for other in [genes[0], genes[1], genes[3]]:
            genome.add_gene(Gene(Connection(genes[4].allele.id, other.allele.id)))

        g1 = genome.get_phenome()
        g2 = genome.copy().get_phenome()

        x = [1, 1, 1]
        self.assertEqual(g1.compute(x), g2.compute(x))

if __name__ == '__main__':
    random.seed(42)

    unittest.main()
    