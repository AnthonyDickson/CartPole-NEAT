import random


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
        copy.is_recurrent = self.is_recurrent

        return copy

    def to_json(self):
        """Encode the connection as JSON.

        Returns: the connection encoded as JSON.
        """
        return dict(
            origin_id=self.origin_id,
            target_id=self.target_id,
            id=self.id,
            weight=self.weight
        )

    @staticmethod
    def from_json(config):
        """Load a node object from JSON.

        Arguments:
            config: the JSON dictionary loaded from file.

        Returns: a connection object.
        """
        connection = Connection(config['origin_id'], config['target_id'])
        connection.id = config['id']
        connection.weight = config['weight']

        return connection

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
