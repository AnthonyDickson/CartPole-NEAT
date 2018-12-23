import random
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

    @staticmethod
    def all():
        """Get a list of all activation functions.

        Returns: a list of references to all activation functions.
        """
        return [Activations.identity, Activations.relu, Activations.sigmoid,
                Activations.tanh, Activations.softmax]


class Node:
    """A node in a neural network computation graph."""
    count = 0  # a count of unique nodes

    def __init__(self, activation=Activations.identity):
        self.output = 0
        self.prev_output = 0
        self._bias = random.gauss(0, 1)
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
        copy._bias = self._bias
        copy.activation = self.activation

        return copy

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, value):
        self._bias = value

    def to_json(self):
        """Encode the node as JSON.

        Returns: the node encoded as a dictionary.
        """
        return dict(
            activation=self.activation.__name__,
            bias=self.bias,
            id=self.id,
            type=self.__class__.__name__
        )

    @staticmethod
    def from_json(config):
        """Load a node object from JSON.

        Arguments:
            config: the JSON dictionary loaded from file.

        Returns: a node object.
        """

        if config['type'] == Sensor.__name__:
            node = Sensor()
        elif config['type'] == Hidden.__name__:
            node = Hidden()
        elif config['type'] == Output.__name__:
            node = Output()
        else:
            raise ValueError('%s is not a supported node type.' %
                             config['type'])

        for activation in Activations.all():
            if config['activation'] == activation.__name__:
                node.activation = activation

                break
        else:
            raise ValueError('%s is not a supported activation.' %
                             config['activation'])

        node.bias = config['bias']
        node.id = config['id']

        return node

    def __str__(self):
        return 'Node_%d' % self.id

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id


# Create distinct node types so we can distinguish them later.
class Sensor(Node):
    """A sensor node (or input node) in a neural network computation graph."""

    def __init__(self):
        super().__init__(Activations.identity)

        self._bias = 0

    @property
    def bias(self):
        # It doesn't make sense to have a bias on the input node.
        return 0

    @bias.setter
    def bias(self, value):
        # The bias on an input node should be immutable.
        return

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
