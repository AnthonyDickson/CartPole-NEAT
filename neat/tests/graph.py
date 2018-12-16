import random
import unittest

from neat.graph import Sensor, Hidden, Output, Connection, Graph, GraphNotCompiledError, InvalidGraphError, InvalidGraphInputError

class GraphUnitTest(unittest.TestCase):
    def test_compile_raises_error(self):
        g = Graph()
        self.assertRaises(InvalidGraphError, g.compile)

        s1 = Sensor()
        g.add_node(s1)   
        
        self.assertRaises(InvalidGraphError, g.compile)
        
        o1 = Output()
        g.add_node(o1)
        
        self.assertRaises(InvalidGraphError, g.compile)

    def test_comutation(self):
        g = Graph()
        nodes = [Sensor(), Sensor(), Sensor(), Output(), Hidden()]
        g.add_nodes(nodes)

        for other in [nodes[0], nodes[2], nodes[4]]:
            g.add_input(nodes[3].id, other.id)

        for other in [nodes[0], nodes[1], nodes[3]]:
            g.add_input(nodes[4].id, other.id)

        g.compile()

        x = [1, 1, 1]
        g.compute(x)
        g.compute(x)

        g.disable_input(4, 3)
        g.compile()
        g.compute(x)

    def test_copy(self):
        g1 = Graph()
        nodes = [Sensor(), Sensor(), Sensor(), Output(), Hidden()]
        g1.add_nodes(nodes)

        for other in [nodes[0], nodes[2], nodes[4]]:
            g1.add_input(nodes[3].id, other.id)

        for other in [nodes[0], nodes[1], nodes[3]]:
            g1.add_input(nodes[4].id, other.id)


        g1.compile()
        g2 = g1.copy()

        x = [1, 1, 1]
        self.assertEqual(g1.compute(x), g2.compute(x))

if __name__ == '__main__':
    random.seed(42)

    unittest.main()
    