from graph import Connection


class Gene:
    """Represents a single gene of a creature.

    This stub is here simply to provide a logical hierarchy for NodeGene and
    ConnectionGene.
    """
    pass


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

    def __str__(self):
        return 'Node_gene(%s)' % self.node

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.node == other.node

    def __hash__(self):
        # Use negative node id to avoid collisions with connection genes.
        return -self.node.id


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

    def __str__(self):
        return 'Connection_Gene_%d(%s)' % (self.innovation_number,
                                           self.connection) + \
               ' (disabled)' if not self.is_enabled else ''

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.connection == other.connection and \
               self.innovation_number == other.innovation_number

    def __hash__(self):
        return self.innovation_number
