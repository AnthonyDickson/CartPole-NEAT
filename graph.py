from collections import defaultdict
from enum import Enum
from math import exp
import random

import numpy as np

class Activations:
    @staticmethod
    def identity(x):
        return x

    @staticmethod
    def relu(x):
        return max(0, x)

    @staticmethod
    def sigmoid(x):
        1 / (1 + exp(-x))

    @staticmethod
    def tanh(x):
        return (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    @staticmethod
    def softmax(z):
        z_exp = np.exp(z)
        return z_exp / z_exp.sum()

class Node:
    node_count = 0

    def __init__(self, activation=Activations.identity):
        self.output = 0
        self.prev_output = 0
        self.bias = random.gauss(0, 1)
        self.activation = activation
        self.inputs = []

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
        self.nodes = [Sensor(), Sensor(), Sensor(), Output(), Hidden(activation=Activations.sigmoid)]
        self.sensors = self.nodes[:3]
        self.hidden = [self.nodes[4]]
        self.outputs = [self.nodes[3]]

        self.verbosity = verbosity

        node = self.nodes[3]

        for other in [0, 2, 4]:
            self.add_input(node, other)

        node = self.nodes[4]

        for other in [0, 1, 3]:
            self.add_input(node, other)

        self.compile()

    def compile(self):
        for output in self.outputs:
            self._mark_recurrent_inputs(output, set())

    def _mark_recurrent_inputs(self, curr, visited):
        visited.add(curr)

        for i in curr.inputs:
            if i.target in visited:
                i.is_recurrent = True
            else:
                self._mark_recurrent_inputs(i.target, visited.copy())

    def add_input(self, node, other):
        node.inputs.append(Connection(node, other))

    def disable_input(self, node, other):
        for i in node.inputs:
            if i.target == other:
                i.is_enabled = False
                self.print('Disabling input from %s to %s.' % (i.origin, i.target))


    def compute(self, x):
        assert len(x) == len(self.sensors)

        for node in self.nodes:
            node.prev_output = node.output

        for x, sensor in zip(x, self.sensors):
            sensor.output = x

        network_output = []

        for output in self.outputs:
            network_output.append(self._compute_output(output))

        return Activations.softmax(network_output)

    def _compute_output(self, node, level=0):
        self.print('%s%s Computing...' % ('\t' * level, node))

        node_output = node.output if isinstance(node, Sensor) else node.bias

        for i in node.inputs:
            if not i.is_enabled:
                self.print('%s%s Input from this node is disabled!' % ('\t' * (level + 1), i.target))
            elif i.is_recurrent:
                node_output += i.weight * i.target.prev_output
                self.print('%s%s Output (recurrent): %f' % ('\t' * (level + 1), i.target, i.target.prev_output))
            else:
                node_output += i.weight * self._compute_output(i.target, level=level + 1)

        node.output = node_output
        self.print('%s%s Output: %f' % ('\t' * level, node, node.output))

        return node_output

    def print_connections(self):        
        for node in self.nodes:
            for i in node.inputs:
                print(i)

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

    g.disable_input(g.nodes[4], g.nodes[3])

    g.compute(x)
    