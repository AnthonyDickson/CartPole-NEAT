from collections import defaultdict
from enum import Enum
from math import exp
import random

import numpy as np

class Activations:
    """Contains various activation functions."""
    @staticmethod
    def identity(x):
        return x

    @staticmethod
    def relu(x):
        return max(0, x)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + exp(-x))

    @staticmethod
    def tanh(x):
        return (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    @staticmethod
    def softmax(z):
        z_exp = np.exp(z)
        return z_exp / z_exp.sum()

class Node:
    """A node in a neural network computation graph."""
    count = 0

    def __init__(self, activation=Activations.identity):
        self.output = 0
        self.prev_output = 0
        self.bias = random.gauss(0, 1)
        self.activation = activation
        self.inputs = []

        self.id = Node.count
        Node.count += 1

    def __str__(self):
        return 'Node_%d' % self.id

# Create distinct node types so we can distinguish them later.
class Sensor(Node):
    """A sensor node (or input node) in a neural network computation graph."""
    def __init__(self):
        super().__init__(Activations.identity)

class Hidden(Node):
    """A hidden node in a neural network computation graph."""
    def __init__(self, activation=Activations.sigmoid):
        super().__init__(activation)

class Output(Node):
    """An output node in a neural network computation graph."""
    pass

class Connection:
    """A connection between two nodes in a neural network computation graph."""
    count = 0

    def __init__(self, origin, target):
        self.origin = origin
        self.target = target
        self.weight = random.gauss(0, 1)
        self.is_enabled = True
        self.is_recurrent = False

        self.id = Connection.count
        Connection.count += 1

    def __str__(self):
        return '{} -> {}'.format(self.origin, self.target) + (' (recurrent)' if self.is_recurrent else '')

class Verbosity(Enum):
    SILENT = 0
    MINIMAL = 1
    FULL = 2    

class InvalidGraphError(Exception):
    pass

class GraphNotCompiledError(Exception):
    pass

class InvalidGraphInputError(Exception):
    pass

class Graph:
    """A computation graph for arbitrary neural networks that allows recurrent connections."""

    def __init__(self, verbosity=Verbosity.MINIMAL):        
        """Initialise the graph.

        Arguments:
            verbosity: how much should be printed to console.
        """
        self.nodes = []
        self.sensors = []
        self.hidden = []
        self.outputs = []

        self.verbosity = verbosity
        self.is_compiled = False

    def compile(self):
        """Make sure the graph is valid and prepare it for computation.
        
        Throws: InvalidGraphError
        """
        if len(self.sensors) == 0:
            raise InvalidGraphError('Graph needs at least one sensor (input) node.')

        if len(self.outputs) == 0:
            raise InvalidGraphError('Graph needs at least one output node.')

        has_path_to_input = False

        for output in self.outputs:
            self._mark_recurrent_inputs(output)            
            has_path_to_input |= self._has_path_to_input(output)

        if not has_path_to_input:
            raise InvalidGraphError('Graph needs at least one sensor (input) to be connected to an output.')

        self.is_compiled = True

    def _mark_recurrent_inputs(self, curr, visited=set()):
        """Mark recurrent connections (i.e. cycles) in the graph.

        Arguments:
            curr: the current node that is being evaluated. This should initially be set to a terminal node (such as an output node).
            visited: the list of visited nodes in the search. Can also be thought of the current node's ancestor nodes. Initially this should be an empty set.
        """
        visited.add(curr)

        for node_input in curr.inputs:
            if node_input.target in visited:
                node_input.is_recurrent = True
            else:
                self._mark_recurrent_inputs(node_input.target, visited.copy())

    def _has_path_to_input(self, node, visited=set()):
        """Check if the given node has a path to the input.

        This is generally needed to check the the graph has at least one input connected to an output.

        Arguments:
            node: the node that should be checked for a path to an input node.
            visited: the list of visited nodes in the search. Initially this should be an empty set.

        Returns: True if a path exists to an input node, False otherwise.
        """
        visited.add(node)

        for node_input in node.inputs:
            if not node_input.target in visited: 
                if self._has_path_to_input(node_input.target, visited.copy()):
                    return True

        return isinstance(node, Sensor)

    def add_node(self, node):
        """Add a node to the graph.

        Arguments:
            node: The node to be added.
        """
        self.nodes.append(node)

        if isinstance(node, Sensor):
            self.sensors.append(node)
        elif isinstance(node, Output):
            self.outputs.append(node)

    def add_nodes(self, nodes):
        """Helper function to add a list of nodes to the graph.

        Arguments:
            nodes: a list of nodes that are to be added to the graph.
        """
        for node in nodes:
            self.add_node(node)

    def add_input(self, node, other):
        """Add an input (form a connection) to a node. 

        Arguments:
            node: the node that will receive the input.
            other: the node that will provide the input.
        """
        node.inputs.append(Connection(node, other))

    def disable_input(self, node, other):
        """Disable an input of a node.

        Does not remove the input connection.

        Arguments:
            node: the node that receives the input.
            other: the node that provides the input.
        """
        for node_input in node.inputs:
            if node_input.target == other:
                node_input.is_enabled = False
                self.print('Disabling input from %s to %s.' % (node_input.origin, node_input.target))

    def compute(self, x):
        """Compute the output of the neural network graph.

        Arguments:
            x: the input vector (one dimensional).

        Returns: the softmax output of the neural network graph.
        """
        if not self.is_compiled:
            raise GraphNotCompiledError('The graph must be compiled before being used.')

        if len(x) != len(self.sensors):
            raise InvalidGraphInputError('The input dimenions do not match the number of input nodes in the graph.')

        for node in self.nodes:
            node.prev_output = node.output

        for x, sensor in zip(x, self.sensors):
            sensor.output = x

        network_output = []

        for output in self.outputs:
            network_output.append(self._compute_output(output))

        output = Activations.softmax(network_output)
        self.print('Network output: %s' % output)

        return output

    def _compute_output(self, node, level=0):
        """Compute the output of a node.

        Arguments:
            node: the node whose output should be computed.
            level: the level, or depth, the current node relative to the starting node (typically an output node).

        Returns: the output of the node.
        """
        self.print('%s%s Computing...' % ('\t' * level, node))

        node_output = node.output if isinstance(node, Sensor) else node.bias

        for node_input in node.inputs:
            if not node_input.is_enabled:
                self.print('%s%s Input from this node is disabled!' % ('\t' * (level + 1), node_input.target))
            elif node_input.is_recurrent:
                node_output += node_input.weight * node_input.target.prev_output
                self.print('%s%s Output (recurrent): %f' % ('\t' * (level + 1), node_input.target, node_input.target.prev_output))
            else:
                node_output += node_input.weight * self._compute_output(node_input.target, level=level + 1)

        node.output = node.activation(node_output)
        self.print('%s%s Output: %f' % ('\t' * level, node, node.output))

        return node.output

    def print_connections(self):   
        """Print the connections (inputs) of every node in the graph."""     
        for node in self.nodes:
            for node_input in node.inputs:
                print(node_input)

    def print(self, msg, verbosity=Verbosity.MINIMAL):
        """Print a message whose visibility is controlled by the verbosity of the message and the graphs verbosity setting.
        
        Arguments:
            msg: The string to print.
            verbosity: The verbosity of the message to print.
        """     
        if self.verbosity.value >= verbosity.value:
            print(msg)

if __name__ == '__main__':
    random.seed(42)

    g = Graph()
    nodes = [Sensor(), Sensor(), Sensor(), Output(), Hidden()]
    g.add_nodes(nodes)

    node = nodes[3]

    for other in [0, 2, 4]:
        g.add_input(node, nodes[other])

    node = nodes[4]

    for other in [0, 1, 3]:
        g.add_input(node, nodes[other])

    g.compile()
    g.print_connections()

    x = [1, 1, 1]
    g.compute(x)
    g.compute(x)

    g.disable_input(g.nodes[4], g.nodes[3])

    g.compute(x)
    