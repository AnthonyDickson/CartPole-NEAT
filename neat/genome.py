"""Implements a basic model of genes, genomes (genotypes), and phenotypes of creatures in NEAT."""

from neat.graph import Graph

class Gene:
    """Represents a single gene of a creature.

    This stub is here simply to provide a sensible hierarchy for NodeGene and ConnectionGene.
    """
    pass

class NodeGene(Gene):
    """Represents a node gene."""

    def __init__(self, node_type):
        """Create a node gene.

        Arguments:
            node_type: the type of node this gene represents.
        """
        self.node_type = node_type
        self.bias = 0

    def copy(self):
        """Make a copy of this gene.

        Returns: the copy of this gene.
        """
        copy = NodeGene(self.node_type)

        return copy

    def __str__(self):
        return 'Gene_Node(%d)' % self.node_type.__name__

    def __eq__(self, other):
        try:
            return self.__name__ == other.__name__
        except AttributeError:
            return False

class ConnectionGene(Gene):
    """Represents a connection gene."""
    pool = {}

    def __init__(self, input_node_id, output_node_id):
        """Create a connection gene.

        Arguments:
            input_node_id: the id of the node that provides the input.
            output_node_id: the id of the node that receives the input.
        """
        self.input_node_id = input_node_id
        self.output_node_id = output_node_id
        self.weight = 0
        self.is_disabled = False

        try:
            self.innovation_number = ConnectionGene.pool[self]
        except KeyError:
            self.innovation_number = len(ConnectionGene.pool)
            ConnectionGene.pool[self] = self.innovation_number

    def copy(self):
        """Make a copy of this gene.

        Returns: the copy of this gene.
        """
        copy = ConnectionGene(self.input_node_id, self.output_node_id)

        copy.is_disabled = self.is_disabled
        copy.innovation_number = self.innovation_number

        return copy

    def __str__(self):
        return 'Gene_Connection(%s->%s)' % (self.input_node_id, self.output_node_id)

    def __eq__(self, other):
        return self.input_node_id == other.input_node_id and \
            self.output_node_id == other.output_node_id

    def __hash__(self):
        hash_code = 7
        hash_code += hash_code * self.input_node_id % 17
        hash_code += hash_code * self.output_node_id % 37

        return hash_code

class Genome():
    """Represents a creature's genome (a set of genes)."""

    def __init__(self):
        self.nodes = []
        self.connections = []

    def copy(self):
        """Make a copy of a genome.

        Returns: The copy of the genome.
        """
        copy = Genome()
        copy.nodes = [node.copy() for node in self.nodes]
        copy.connections = [connection.copy() for connection in self.connections]

        return copy

    def add_gene(self, gene):
        """Add a gene to the genome.

        The innovation number of the gene is also set in this method.

        Arguments:
            gene: The gene to be added to the genome.
        """
        if isinstance(gene, NodeGene):
            self.nodes.append(gene)
        else:
            self.connections.append(gene)

    def add_genes(self, genes):
        """Add a list of genes to the genome.

        Arguments:
            genes: a list of Gene objects that are to be added.
        """
        for gene in genes:
            self.add_gene(gene)

class Phenotype(Graph):
    """A phenotype, or physical expression, of a genome (genotype)."""

    def __init__(self, genome):
        """Generate the phenotype, or physical expression, of a genome.

        Arguments:
            genome: the genome to express.
        """
        super().__init__()

        for gene in genome.nodes:
            node = gene.node_type()
            node.id = genome.nodes.index(gene)

            self.add_node(node)

        for gene in genome.connections:
            if gene.is_disabled:
                continue

            self.add_input(gene.output_node_id, gene.input_node_id)

        self.compile()
