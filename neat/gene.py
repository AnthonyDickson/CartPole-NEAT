from neat.connection import Connection
from neat.node import Node


class Gene:
    """Represents a single gene of a creature."""

    @property
    def alignment_key(self):
        """The key by which the gene should be aligned.

        This acts as a numbering system for genes.
        """
        raise NotImplementedError

    def combine_by_average(self, other):
        """Combine two genes by taking the average of their distinct values.

        This is typically done between two aligned genes.

        Arguments:
            other: The other gene to combine with.

        Returns: a single new gene, where their 'traits' have been averaged.
        """
        raise NotImplementedError

    def to_json(self):
        """Encode the gene as JSON.

        Returns: the JSON encoded genes.
        """
        raise NotImplementedError

    @staticmethod
    def from_json(config):
        """Load a gene object from JSON.

        Arguments:
            config: the JSON dictionary loaded from file.

        Returns: a gene object.
        """
        raise NotImplementedError


class NodeGene(Gene):
    """Represents a node gene."""

    def __init__(self, node):
        """Create a node gene.

        Arguments:
            node: the node this gene represents.
        """
        self.node = node

    def copy(self):
        """Make a copy of this gene.

        Returns: the copy of this gene.
        """
        copy = NodeGene(self.node.copy())

        return copy

    @property
    def alignment_key(self):
        return self.node.id

    @property
    def bias(self):
        return self.node.bias

    @bias.setter
    def bias(self, value):
        self.node.bias = value

    def combine_by_average(self, other):
        new_gene = self.copy()

        new_gene.bias = 0.5 * (self.bias + other.bias)

        return new_gene

    def to_json(self):
        return dict(node=self.node.to_json())

    @staticmethod
    def from_json(config):
        node = Node.from_json(config['node'])
        gene = NodeGene(node)

        return gene

    def __str__(self):
        return 'Node_gene(%s)' % self.node

    def __eq__(self, other):
        return self.node == other.node

    def __hash__(self):
        # Use negative node id to avoid collisions with connection genes.
        return -self.node.id

    def __lt__(self, other):
        return self.node.id < other.node.id


class ConnectionGene(Gene):
    """Represents a connection gene."""
    pool = {}

    def __init__(self, origin_id=None, target_id=None):
        """Create a connection gene.

        Creates a empty connection gene if either origin_id or target_id
        are set to None.

        Arguments:
            origin_id: the id of the node that receives the input.
            target_id: the id of the node that provides the input.
        """
        self.is_enabled = True

        if origin_id is None or target_id is None:
            self.connection = None
            self.innovation_number = None

            return

        self.connection = Connection(origin_id, target_id)

        try:
            self.innovation_number = ConnectionGene.pool[self.connection]
        except KeyError:
            self.innovation_number = len(ConnectionGene.pool) + 1
            ConnectionGene.pool[self.connection] = self.innovation_number

    def copy(self):
        """Make a copy of this gene.

        Returns: the copy of this gene.
        """
        copy = ConnectionGene()
        copy.connection = self.connection.copy()
        copy.is_enabled = self.is_enabled
        copy.innovation_number = self.innovation_number

        return copy

    @property
    def alignment_key(self):
        return self.innovation_number

    @property
    def weight(self):
        return self.connection.weight

    @weight.setter
    def weight(self, value):
        self.connection.weight = value

    def combine_by_average(self, other):
        new_gene = self.copy()

        new_gene.weight = 0.5 * (self.weight + other.weight)

        return new_gene

    def to_json(self):
        return dict(
            connection=self.connection.to_json(),
            innovation_number=self.innovation_number,
            is_enabled=self.is_enabled
        )

    @staticmethod
    def from_json(config):
        gene = ConnectionGene()
        gene.connection = Connection.from_json(config['connection'])
        gene.innovation_number = config['innovation_number']
        gene.is_enabled = config['is_enabled']

        return gene

    def __str__(self):
        return 'Connection_Gene_%d(%s)' % (self.innovation_number,
                                           self.connection) + \
               ' (disabled)' if not self.is_enabled else ''

    def __eq__(self, other):
        return self.connection == other.connection and \
               self.innovation_number == other.innovation_number

    def __hash__(self):
        return self.innovation_number

    def __lt__(self, other):
        return self.innovation_number < other.innovation_number
