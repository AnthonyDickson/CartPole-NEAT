import json
import random
import unittest

from neat.gene import NodeGene, ConnectionGene
from neat.node import Hidden


class GeneUnitTest(unittest.TestCase):
    def test_node_gene_json(self):
        """Test whether a node gene can be saved to and loaded from JSON."""
        ng = NodeGene(Hidden())
        dump = json.dumps(ng.to_json())
        ng_load = NodeGene.from_json(json.loads(dump))

        self.assertEqual(ng, ng_load)

    def test_connection_gene_json(self):
        """Test whether a connection gene can be saved to and loaded from JSON.
        """
        cg = ConnectionGene(0, 1)
        dump = json.dumps(cg.to_json())
        cg_load = ConnectionGene.from_json(json.loads(dump))

        self.assertEqual(cg, cg_load)
        self.assertEqual(cg.is_enabled, cg_load.is_enabled)


if __name__ == '__main__':
    random.seed(42)

    unittest.main()
