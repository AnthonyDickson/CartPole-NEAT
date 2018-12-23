import json
import random
import unittest

from neat.connection import Connection
from neat.graph import Graph, InvalidGraphError
from neat.node import Sensor, Hidden, Output, Node


# noinspection PyMethodMayBeStatic
class GraphUnitTest(unittest.TestCase):
    @staticmethod
    def test_graph():
        g = Graph()
        nodes = [Sensor(), Sensor(), Sensor(), Output(), Hidden()]
        g.add_nodes(nodes)

        for other in [nodes[0], nodes[2], nodes[4]]:
            g.add_input(nodes[3].id, other.id)

        for other in [nodes[0], nodes[1], nodes[3]]:
            g.add_input(nodes[4].id, other.id)

        g.compile()

        return g

    def test_compile_raises_error(self):
        g = Graph()
        self.assertRaises(InvalidGraphError, g.compile)

        s1 = Sensor()
        g.add_node(s1)

        self.assertRaises(InvalidGraphError, g.compile)

        o1 = Output()
        g.add_node(o1)

        self.assertRaises(InvalidGraphError, g.compile)

    def test_computation(self):
        g = GraphUnitTest.test_graph()

        x = [1, 1, 1]
        g.compute(x)
        g.compute(x)

    def test_copy(self):
        g1 = GraphUnitTest.test_graph()

        g1.compile()
        g2 = g1.copy()

        x = [1, 1, 1]
        self.assertEqual(g1.compute(x), g2.compute(x))

    def test_node_json(self):
        """Test whether a node can be saved to and loaded from JSON."""
        n = Sensor()

        out_file = json.dumps(n.to_json())
        n_load = Node.from_json(json.loads(out_file))

        self.assertEqual(n, n_load)
        self.assertEqual(n.bias, n_load.bias)
        self.assertEqual(n.activation, n_load.activation)

    def test_connection_json(self):
        """Test whether a connection can be saved to and loaded from JSON."""
        c = Connection(1, 0)

        out_file = json.dumps(c.to_json())

        c_load = Connection.from_json(json.loads(out_file))

        self.assertEqual(c, c_load)
        self.assertEqual(c.id, c_load.id)
        self.assertEqual(c.weight, c_load.weight)

    def test_graph_json(self):
        """Test whether a graph can be  saved to and loaded from JSON."""
        g = GraphUnitTest.test_graph()

        x = [1, 2, 3]
        g_out = g.compute(x)

        out_file = json.dumps(g.to_json())
        g_load = Graph.from_json(json.loads(out_file))

        self.assertEqual(g_out, g_load.compute(x))


if __name__ == '__main__':
    random.seed(42)

    unittest.main()
