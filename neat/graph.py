from collections import defaultdict
from enum import Enum
from math import exp
import random
import unittest

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

        self.id = Node.count
        Node.count += 1

    def copy(self):
        """Make a copy of a node.
        
        Returns: a copy of the node.
        """
        copy = self.__class__()

        copy.bias = self.bias
        copy.activation = self.activation

        return copy

    def __str__(self):
        return 'Node_%d' % self.id

# Create distinct node types so we can distinguish them later.
class Sensor(Node):
    """A sensor node (or input node) in a neural network computation graph."""
    
    def __init__(self):
        super().__init__(Activations.identity)

    def __str__(self):
        return 'Sensor_%d' % self.id

class Hidden(Node):
    """A hidden node in a neural network computation graph."""

    def __init__(self, activation=Activations.sigmoid):
        super().__init__(activation)

    def __str__(self):
        return 'Hidden_%d' % self.id

class Output(Node):
    """An output node in a neural network computation graph."""
    def __init__(self, activation=Activations.sigmoid):
        super().__init__(activation)


    def __str__(self):
        return 'Output_%d' % self.id

class Connection:
    """A connection between two nodes in a neural network computation graph."""
    count = 0

    def __init__(self, origin_id, target_id):
        self.origin_id = origin_id
        self.target_id = target_id
        self.weight = random.gauss(0, 1)
        self.is_enabled = True
        self.is_recurrent = False

        self.id = Connection.count
        Connection.count += 1

    def copy(self):
        """Make a copy of a connection.

        Returns: the copy of the connection.
        """
        copy = Connection(self.origin_id, self.target_id)

        copy.weight = self.weight
        copy.is_enabled = self.is_enabled

        return copy

    def __str__(self):
        return '{} -> {}'.format(self.origin_id, self.target_id) + (' (recurrent)' if self.is_recurrent else '')

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

    def __init__(self, verbosity=Verbosity.SILENT):        
        """Initialise the graph.

        Arguments:
            verbosity: how much should be printed to console.
        """
        self.nodes = []
        self.sensors = []
        self.hidden = []
        self.outputs = []
        self.connections = defaultdict(lambda: [])

        self.verbosity = verbosity
        self.is_compiled = False

    def copy(self):
        """Make a copy of a graph.

        Returns: the copy of the graph.
        """
        copy = Graph()
        
        for i, node in enumerate(self.nodes):
            copy.add_node(node.copy())

            for connection in self.connections[i]:
                copy.connections[i].append(connection.copy())
    
        return copy

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

    def _mark_recurrent_inputs(self, node_id, visited=set()):
        """Mark recurrent connections (i.e. cycles) in the graph.

        Arguments:
            node_id: the id (position in the nodes list) of the current node that is being evaluated. This should initially be set to a terminal node (such as an output node).
            visited: the list of visited nodes in the search. Can also be thought of the current node's ancestor nodes. Initially this should be an empty set.
        """
        visited.add(node_id)

        for input_connection in self.connections[node_id]:
            if input_connection.target_id in visited:
                input_connection.is_recurrent = True
            else:
                self._mark_recurrent_inputs(input_connection.target_id, visited.copy())

    def _has_path_to_input(self, node_id, visited=set()):
        """Check if the given node has a path to the input.

        This is generally needed to check the the graph has at least one input connected to an output.

        Arguments:
            node_id: the id of the node that should be checked for a path to an input node.
            visited: the list of visited nodes in the search. Initially this should be an empty set.

        Returns: True if a path exists to an input node, False otherwise.
        """
        visited.add(node_id)

        for node_input in self.connections[node_id]:
            if not node_input.target_id in visited: 
                if self._has_path_to_input(node_input.target_id, visited.copy()):
                    return True

        return isinstance(self.nodes[node_id], Sensor)

    def add_node(self, node):
        """Add a node to the graph.

        Arguments:
            node: The node to be added.
        """
        self.nodes.append(node)

        if isinstance(node, Sensor):
            self.sensors.append(len(self.nodes) - 1)
        elif isinstance(node, Output):
            self.outputs.append(len(self.nodes) - 1)

    def add_nodes(self, nodes):
        """Helper function to add a list of nodes to the graph.

        Arguments:
            nodes: a list of nodes that are to be added to the graph.
        """
        for node in nodes:
            self.add_node(node)

    def add_input(self, node_id, other_id):
        """Add an input (form a connection) to a node. 

        Arguments:
            node_id: the id of the node that will receive the input.
            other_id: the id of the node that will provide the input.
        """
        self.connections[node_id].append(Connection(node_id, other_id))

    def disable_input(self, node_id, other_id):
        """Disable an input of a node.

        Does not remove the input connection.

        Arguments:
            node_id: the id of the node that receives the input.
            other_id: the id of the node that provides the input.
        """
        for node_input in self.connections[node_id]:
            if node_input.target_id == other_id:
                node_input.is_enabled = False
                self.print('Disabling input from %s to %s.' % (node_input.origin_id, node_input.target_id))

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
            self.nodes[sensor].output = x

        network_output = []

        for output in self.outputs:
            network_output.append(self._compute_output(output))

        output = Activations.softmax(network_output) if len(self.outputs) > 0 else network_output
        self.print('Network output: %s' % output)

        return output

    def _compute_output(self, node_id, level=0):
        """Compute the output of a node.

        Arguments:
            node_id: the id of the node whose output should be computed.
            level: the level, or depth, the current node relative to the starting node (typically an output node).

        Returns: the output of the node.
        """
        self.print('%s%s Computing...' % ('\t' * level, node_id))

        node = self.nodes[node_id]

        node_output = node.output if isinstance(node, Sensor) else node.bias

        for input_connection in self.connections[node_id]:
            target = self.nodes[input_connection.target_id]

            if not input_connection.is_enabled:
                self.print('%s%s Input from this node is disabled!' % ('\t' * (level + 1), input_connection.target_id))
            elif input_connection.is_recurrent:
                node_output += input_connection.weight * target.prev_output
                self.print('%s%s Output (recurrent): %f' % ('\t' * (level + 1), input_connection.target_id, target.prev_output))
            else:
                node_output += input_connection.weight * self._compute_output(input_connection.target_id, level=level + 1)

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

