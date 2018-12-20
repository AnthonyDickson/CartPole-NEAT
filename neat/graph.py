"""Describes a computation graph."""

import random
from collections import defaultdict
from enum import Enum
from math import exp

import numpy as np


class Activations:
    """Contains various activation functions."""

    @staticmethod
    def identity(x):
        """The identity activation function.

        Arguments:
            x: The value to modify.

        Returns: x unchanged.
        """
        return x

    @staticmethod
    def relu(x):
        """The Rectified Linear Unit activation function.

        Arguments:
            x: The value to modify.

        Returns: x if x > 0, 0 otherwise.
        """
        return max(0, x)

    @staticmethod
    def sigmoid(x):
        """The logistic activation function.

        Arguments:
            x: The value to modify.

        Returns: 1 / (1 + exp(-x)).
        """
        # This prevents math range errors with large negative numbers.
        if x < 0:
            return 1 - 1 / (1 + exp(x))

        return 1 / (1 + exp(-x))

    @staticmethod
    def tanh(x):
        """The hyperbolic tangent activation function.

        Essentially a scaled and shifted logistic function.

        Arguments:
            x: The value to modify.

        Returns: (exp(x) - exp(-x)) / (exp(x) + exp(-x)).
        """
        return (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    @staticmethod
    def softmax(x):
        """The softmax activation function..

        Arguments:
            x: The list of values to modify.

        Returns: a list of numbers in the interval [0, 1) representing a
                 probability distribution.
        """
        z_exp = np.exp(x)
        return z_exp / z_exp.sum()


class Node:
    """A node in a neural network computation graph."""
    count = 0  # a count of unique nodes

    def __init__(self, activation=Activations.identity):
        self.output = 0
        self.prev_output = 0
        self.bias = random.gauss(0, 1)
        self.activation = activation

        Node.count += 1
        self.id = Node.count

    def copy(self):
        """Make a copy of a node.

        Returns: a copy of the node.
        """
        copy = self.__class__()
        # copies of nodes are not unique and therefore not counted.
        Node.count -= 1

        copy.id = self.id
        copy.bias = self.bias
        copy.activation = self.activation

        return copy

    def __str__(self):
        return 'Node_%d' % self.id

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id


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
    count = 0  # a count of unique nodes

    def __init__(self, origin_id, target_id):
        """Create a connection between nodes.

        Arguments:
            origin_id: The id of the node that receives the input.
            target_id: The id of the node that provides the input.
        """
        self.origin_id = origin_id
        self.target_id = target_id
        self.weight = random.gauss(0, 1)
        self.is_enabled = True
        self.is_recurrent = False

        Connection.count += 1
        self.id = Connection.count

    def copy(self):
        """Make a copy of a connection.

        Returns: the copy of the connection.
        """
        copy = Connection(self.origin_id, self.target_id)
        # copies of connections are not unique and therefore not counted.
        Connection.count -= 1
        copy.id = self.id
        copy.weight = self.weight
        copy.is_enabled = self.is_enabled
        copy.is_recurrent = self.is_recurrent

        return copy

    def __str__(self):
        return 'Connection_{}->{}'.format(self.origin_id, self.target_id) + \
               (' (recurrent)' if self.is_recurrent else '')

    def __eq__(self, other):
        return self.origin_id == other.origin_id and \
               self.target_id == other.target_id

    def __hash__(self):
        hash_code = 7
        hash_code += hash_code * self.origin_id % 17
        hash_code += hash_code * self.target_id % 37

        return hash_code


class Verbosity(Enum):
    """An enum capturing different verbosity levels for logging."""
    SILENT = 0
    MINIMAL = 1
    FULL = 2


class InvalidGraphError(Exception):
    """An error that occurs when a graph is tried to be compiled but it does
    not have a valid structure.
    """
    pass


class GraphNotCompiledError(Exception):
    """An error that occurs when a graph is tried to be used but has not
    been compiled.
    """
    pass


class InvalidGraphInputError(Exception):
    """An error that occurs when the input to a graph does not match the number
    of input nodes.
    """
    pass


class Graph:
    """A computation graph for arbitrary neural networks that allows recurrent
    connections.
    """

    def __init__(self, verbosity=Verbosity.SILENT):
        """Initialise the graph.

        Arguments:
            verbosity: how much should be printed to console.
        """
        self.nodes = {}
        self.sensors = []
        self.outputs = []
        self.connections = defaultdict(lambda: [])

        self.verbosity = verbosity
        self.is_compiled = False

    def copy(self):
        """Make a copy of a graph.

        Returns: the copy of the graph.
        """
        copy = Graph()

        for node in self.nodes:
            copy.add_node(self.nodes[node].copy())

            for connection in self.connections[node]:
                copy.connections[node].append(connection.copy())

        # If a graph is copied as-is, then it should still be compiled if the
        # original was compiled, and not compiled if the other was not
        # compiled.
        copy.is_compiled = self.is_compiled

        return copy

    def compile(self):
        """Make sure the graph is valid and prepare it for computation.

        Throws: InvalidGraphError
        """
        if not self.sensors:
            raise InvalidGraphError('Graph needs at least one sensor (input)'
                                    'node.')

        if not self.outputs:
            raise InvalidGraphError('Graph needs at least one output node.')

        has_path_to_input = False

        for output in self.outputs:
            self._mark_recurrent_inputs(output)
            has_path_to_input |= self._has_path_to_input(output)

        if not has_path_to_input:
            raise InvalidGraphError('Graph needs at least one sensor (input) '
                                    'to be connected to an output.')

        self.is_compiled = True

    def _mark_recurrent_inputs(self, node_id, visited=None):
        """Mark recurrent connections (i.e. cycles) in the graph.

        Arguments:
            node_id: the id (position in the nodes list) of the current node
                     that is being evaluated. This should initially be set
                     to a terminal node (such as an output node).
            visited: the list of visited nodes in the search. Can also be
                     thought of the current node's ancestor nodes. Initially
                     this should be an empty set.
        """
        if visited is None:
            visited = set()

        visited.add(node_id)

        for input_connection in self.connections[node_id]:
            if input_connection.target_id in visited:
                input_connection.is_recurrent = True
            else:
                self._mark_recurrent_inputs(input_connection.target_id,
                                            visited.copy())

    def _has_path_to_input(self, node_id, visited=None):
        """Check if the given node has a path to the input.

        This is generally needed to check the the graph has at least one
        input connected to an output.

        Arguments:
            node_id: the id of the node that should be checked for a path to
                     an input node.
            visited: the list of visited nodes in the search. Initially this
                     should be an empty set.

        Returns: True if a path exists to an input node, False otherwise.
        """
        if visited is None:
            visited = set()

        visited.add(node_id)

        for node_input in self.connections[node_id]:
            if node_input.target_id not in visited and node_input.is_enabled \
                    and self._has_path_to_input(node_input.target_id,
                                                visited.copy()):
                return True

        return isinstance(self.nodes[node_id], Sensor)

    def add_node(self, node):
        """Add a node to the graph.

        Arguments:
            node: The node to be added.
        """
        self.nodes[node.id] = node

        if isinstance(node, Sensor):
            self.sensors.append(node.id)
        elif isinstance(node, Output):
            self.outputs.append(node.id)

        # Adding a node may break the graph so we force the graph to be
        # compiled again to enforce a re-run of sanity and validity checks.
        self.is_compiled = False

    def add_nodes(self, nodes):
        """Helper function to add a list of nodes to the graph.

        Arguments:
            nodes: a list of nodes that are to be added to the graph.
        """
        for node in nodes:
            self.add_node(node)

    def add_connection(self, connection):
        """Add a connection directly to the graph.

        Arguments:
            connection: The Connection object to be added to the graph.
        """
        self.connections[connection.origin_id].append(connection)

        # Adding a connection may break the graph so we force the graph to be
        # compiled again to enforce a re-run of sanity and validity checks.
        self.is_compiled = False

    def add_input(self, node_id, other_id):
        """Add an input (form a connection) to a node.

        Arguments:
            node_id: the id of the node that will receive the input.
            other_id: the id of the node that will provide the input.
        """
        self.connections[node_id].append(Connection(node_id, other_id))

        # Adding a connection may break the graph so we force the graph to be
        # compiled again to enforce a re-run of sanity and validity checks.
        self.is_compiled = False

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

        # Disabling a connection may break the graph so we force the graph to
        # be compiled again to enforce a re-run of sanity and validity checks.
        self.is_compiled = False

    def compute(self, x):
        """Compute the output of the neural network graph.

        Arguments:
            x: the input vector (one dimensional).

        Returns: the softmax output of the neural network graph.
        """
        if not self.is_compiled:
            raise GraphNotCompiledError('The graph must be compiled before '
                                        'being used, or after a change '
                                        'occurred to the graph structure.')

        if len(x) != len(self.sensors):
            raise InvalidGraphInputError('The input dimensions do not match '
                                         'the number of input nodes in the '
                                         'graph.')

        for node in self.nodes:
            self.nodes[node].prev_output = self.nodes[node].output

        for x, sensor in zip(x, self.sensors):
            self.nodes[sensor].output = x

        network_output = []

        for output in self.outputs:
            network_output.append(self._compute_output(output))

        if len(network_output) == 1:
            return network_output[0]
        else:
            return Activations.softmax(network_output)

    def _compute_output(self, node_id, level=0):
        """Compute the output of a node.

        Arguments:
            node_id: the id of the node whose output should be computed.
            level: the level, or depth, the current node relative to the
                   starting node (typically an output node).

        Returns: the output of the node.
        """
        node = self.nodes[node_id]

        node_output = node.output if node.id in self.sensors else node.bias

        for input_connection in self.connections[node_id]:
            target = self.nodes[input_connection.target_id]

            if not input_connection.is_enabled:
                continue
            elif input_connection.is_recurrent:
                node_output += input_connection.weight * target.prev_output
            else:
                node_output += input_connection.weight * \
                               self._compute_output(input_connection.target_id,
                                                    level=level + 1)

        node.output = node.activation(node_output)

        return node.output

    def print_connections(self):
        """Print the connections (inputs) of every node in the graph."""
        for node in self.nodes:
            for input_connection in self.connections[node]:
                print(input_connection)

    def print(self, msg, format_args=None, verbosity=Verbosity.MINIMAL):
        """Print a message whose visibility is controlled by the verbosity of
        the message and the graphs verbosity setting.

        Arguments:
            msg: The string to print.
            format_args: any arguments needed for string formatting.
            verbosity: The verbosity of the message to print.
        """
        if self.verbosity.value >= verbosity.value:
            if format_args:
                print(msg % format_args)
            else:
                print(msg)
