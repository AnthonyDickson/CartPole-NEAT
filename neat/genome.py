"""Implements a basic model of genes, genomes (genotypes), and phenotypes of creatures in NEAT."""

from collections import OrderedDict

from neat.graph import Node, Sensor, Hidden, Output, Connection, Graph

class Gene:
    """Represents a single gene of a creature.

    A gene may be any type of node (sensor, hidden, or output), or a connection.
    Genes also have an innovation number, which is assigned by the genome.
    """

    # Gene pool keeps track of unique genes, and is used to determine the innovation number of a
    # gene.
    pool = OrderedDict()

    def __init__(self, allele):
        """Create a gene.

        Arguments:
            allele: The thing that this gene will represent. In this implementation, it should
            be a node or connection object.
        """
        self.allele = allele

    def copy(self):
        """Make a copy of this gene.

        Returns: the copy of this gene.
        """
        copy = Gene(self.allele.copy())

        return copy

    @property
    def innovation_number(self):
        """Get the innovation number of the gene.

        The innovation number of a gene is its historical marking and whenever a new, unique gene
        is created (usually through mutation), the innovation number increases. This also means
        that same genes share the same innovation number.

        Returns: the innovation number of the gene.
        """
        try:
            Gene.pool[self] += 1
        except (KeyError, ValueError):
            Gene.pool[self] = 1

        return list(Gene.pool).index(self) + 1

    def __hash__(self):
        """Get the hash of the gene.

        The hash code is in the interval [1, 3] if the gene is a node gene. Otherwise, a hash
        code is generated based on the connection parameters origin_id and target_id.
        The hash function is set up so collisions DO happen. This is bevause the hash function
        is used to check similarity between genes.

        Returns: the generated hash code.
        """
        if isinstance(self.allele, Node):
            return self.allele.id
        else:  # allele must be a connection gene.
            # Set initial hash code for connection to a large number to avoid collisions.
            # This works if the hash of a node gene is just its id (a count of unique nodes)
            # and the number of nodes is always less than this large number.
            hash_code = 123456789
            hash_code += self.allele.origin_id
            hash_code *= self.allele.target_id
            hash_code += 1 if self.allele.is_recurrent else 0

            return hash_code

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __repr__(self):
        return 'Gene %d (%s)' % (self.innovation_number, self.allele)

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
        if isinstance(gene.allele, Node):
            self.nodes.append(gene)
            gene.allele.id = len(self.nodes)
        else:
            self.connections.append(gene)
            gene.allele.id = len(self.connections)

    def add_genes(self, genes):
        """Add a list of genes to the genome.

        Arguments:
            genes: a list of Gene objects that are to be added.
        """
        for gene in genes:
            self.add_gene(gene)

    def get_phenotype(self):
        """Generate the phenotype, or physical expression, of the genome.

        Returns: a compiled computation graph.
        """
        graph = Graph()

        for gene in self.nodes:
            graph.add_node(gene.allele)

        for gene in self.connections:
            graph.add_connection(gene.allele)

        graph.compile()

        return graph
