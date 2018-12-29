import random


class Connection:
    """A connection between two nodes in a neural network computation graph."""
    count = 0  # a count of unique nodes

    def __init__(self, target_id, input_id):
        """Create a connection between nodes.

        Arguments:
            target_id: The id of the node that receives the input.
            input_id: The id of the node that provides the input.
        """
        self.target_id = target_id
        self.input_id = input_id
        self.weight = random.gauss(0, 1)
        self.is_recurrent = False

        Connection.count += 1
        self.id = Connection.count
        self.object_id = id(self)

    def copy(self):
        """Make a copy of a connection.

        Returns: the copy of the connection.
        """
        copy = Connection(self.target_id, self.input_id)
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
            target_id=self.target_id,
            input_id=self.input_id,
            id=self.id,
            object_id=self.object_id,
            weight=self.weight
        )

    @staticmethod
    def from_json(config):
        """Load a node object from JSON.

        Arguments:
            config: the JSON dictionary loaded from file.

        Returns: a connection object.
        """
        connection = Connection(config['target_id'], config['input_id'])
        connection.id = config['id']
        connection.object_id = config['object_id']
        connection.weight = config['weight']

        return connection

    def __str__(self):
        return 'Connection_{}->{}'.format(self.target_id, self.input_id) + \
               (' (recurrent)' if self.is_recurrent else '')

    def __eq__(self, other):
        return self.target_id == other.target_id and \
               self.input_id == other.input_id

    def __hash__(self):
        hash_code = 7
        hash_code += hash_code * self.target_id % 17
        hash_code += hash_code * self.input_id % 37

        return hash_code
