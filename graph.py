from collections import defaultdict
from enum import Enum
import random

class Node:
    node_count = 0

    def __init__(self):
        self.output = 0
        self.prev_output = 0
        self.bias = random.gauss(0, 1)
        self.id = Node.node_count
        Node.node_count += 1

    def __str__(self):
        return 'Node #%d' % self.id

# Create distinct node types so we can distinguish them later.
class Sensor(Node):
    pass

class Hidden(Node):
    pass

class Output(Node):
    pass

class Connection:
    def __init__(self, origin, target):
        self.origin = origin
        self.target = target
        self.weight = random.gauss(0, 1)
        self.is_enabled = True
        self.is_recurrent = False

    def __str__(self):
        return '{} -> {}'.format(self.origin.id, self.target.id) + (' (recurrent)' if self.is_recurrent else '')

class Verbosity(Enum):
    SILENT = 0
    MINIMAL = 1
    FULL = 2    

class Graph:    
    def __init__(self, verbosity=Verbosity.MINIMAL):        
        self.nodes = [Sensor(), Sensor(), Sensor(), Output(), Hidden()]
        self.sensors = self.nodes[:3]
        self.hidden = [self.nodes[4]]
        self.outputs = [self.nodes[3]]
        self.connections = defaultdict(lambda: [])

        self.verbosity = verbosity

        node = self.nodes[3]

        for other in [0, 2, 4]:
            self.connections[node].append(Connection(node, self.nodes[other]))

        node = self.nodes[4]

        for other in [0, 1, 3]:
            self.connections[node].append(Connection(node, self.nodes[other]))

        self.compile()

    def compile(self):
        for output in self.outputs:
            self._mark_recurrent_connections(output, set())

    def _mark_recurrent_connections(self, curr, visited):
        visited.add(curr)

        for connection in self.connections[curr]:
            if connection.target in visited:
                connection.is_recurrent = True
            else:
                self._mark_recurrent_connections(connection.target, visited.copy())

    def disable_connection(self, node, other):
        for connection in self.connections[node]:
            if connection.target == other:
                connection.is_enabled = False
                self.print('Disabling connection between %s and %s.' % (connection.origin, connection.target))


    def compute(self, x):
        assert len(x) == len(self.sensors)

        for node in self.nodes:
            node.prev_output = node.output

        for x, sensor in zip(x, self.sensors):
            sensor.output = x

        network_output = []

        for output in self.outputs:
            network_output.append(self._compute_output(output))

        return network_output

    def _compute_output(self, node, level=0):
        self.print('%s%s Computing...' % ('\t' * level, node))

        node_output = node.output if isinstance(node, Sensor) else node.bias

        for connection in self.connections[node]:
            if not connection.is_enabled:
                self.print('%s%s Connection to this node is disabled!' % ('\t' * (level + 1), connection.target))
            elif connection.is_recurrent:
                node_output += connection.weight * connection.target.prev_output
                self.print('%s%s Output (recurrent): %f' % ('\t' * (level + 1), connection.target, connection.target.prev_output))
            else:
                node_output += connection.weight * self._compute_output(connection.target, level=level + 1)

        node.output = node_output
        self.print('%s%s Output: %f' % ('\t' * level, node, node.output))

        return node_output

    def print_connections(self):        
        for node in self.nodes:
            for connection in self.connections[node]:
                print(connection)

    def print(self, msg, verbosity=Verbosity.MINIMAL):
        if self.verbosity.value >= verbosity.value:
            print(msg)

if __name__ == '__main__':
    random.seed(42)

    g = Graph()
    g.print_connections()

    x = [1, 1, 1]
    g.compute(x)
    g.compute(x)

    g.disable_connection(g.nodes[4], g.nodes[3])

    g.compute(x)
    