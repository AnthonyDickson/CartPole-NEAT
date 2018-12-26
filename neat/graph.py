"""Describes a computation graph."""
from collections import defaultdict
from enum import Enum

from neat.connection import Connection
from neat.node import Node, Sensor, Output, Activations


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
    """A computation graph for arbitrary neural networks that allow recurrent
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
        self.connections = []
        self.connections_dict = defaultdict(lambda: [])

        self.verbosity = verbosity
        self.is_compiled = False

    def copy(self):
        """Make a copy of a graph.

        Returns: the copy of the graph.
        """
        copy = Graph()

        for node in self.nodes:
            copy.add_node(self.nodes[node].copy())

            for connection in self.connections_dict[node]:
                copy.connections_dict[node].append(connection.copy())

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

        for node in self.nodes:
            self.connections += self.connections_dict[node]

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

        for input_connection in self.connections_dict[node_id]:
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

        for node_input in self.connections_dict[node_id]:
            if node_input.target_id not in visited \
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
        self.connections_dict[connection.origin_id].append(connection)

        # Adding a connection may break the graph so we force the graph to be
        # compiled again to enforce a re-run of sanity and validity checks.
        self.is_compiled = False

    def add_input(self, node_id, other_id):
        """Add an input (form a connection) to a node.

        Arguments:
            node_id: the id of the node that will receive the input.
            other_id: the id of the node that will provide the input.
        """
        self.connections_dict[node_id].append(Connection(node_id, other_id))

        # Adding a connection may break the graph so we force the graph to be
        # compiled again to enforce a re-run of sanity and validity checks.
        self.is_compiled = False

    @property
    def recurrent_connections(self):
        return list(filter(lambda c: c.is_recurrent, self.connections))

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

        for input_connection in self.connections_dict[node_id]:
            target = self.nodes[input_connection.target_id]

            if input_connection.is_recurrent:
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
            for input_connection in self.connections_dict[node]:
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

    def __len__(self):
        return len(self.nodes) + len(self.connections)

    def to_json(self):
        """Encode a graph as JSON.

        Returns: the graph encoded as JSON.
        """
        return dict(
            nodes=[node.to_json() for node in self.nodes.values()],
            connections=[connection.to_json()
                         for connection in self.connections],
        )

    @staticmethod
    def from_json(config):
        """Load a graph object from JSON.

        Arguments:
            config: the JSON dictionary loaded from file.

        Returns: a graph object.
        """
        graph = Graph()

        graph.add_nodes([Node.from_json(node)
                         for node in config['nodes']])
        for connection in [Connection.from_json(connection)
                           for connection in config['connections']]:
            graph.add_connection(connection)

        graph.compile()

        return graph
