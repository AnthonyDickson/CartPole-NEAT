"""Implements a basic model of genes, genomes (genotypes), and phenotypes of creatures in NEAT."""

from neat.graph import Graph, Connection


class Gene:
    """Represents a single gene of a creature.

    This stub is here simply to provide a sensible hierarchy for NodeGene and ConnectionGene.
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

    def __eq__(self, other):
        try:
            return self.__name__ == other.__name__
        except AttributeError:
            return False


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
        if origin_id is None or target_id is None:
            self.connection = None
            self.innovation_number = None

            return

        self.connection = Connection(origin_id, target_id)

        try:
            self.innovation_number = ConnectionGene.pool[self.connection]
        except KeyError:
            self.innovation_number = len(ConnectionGene.pool)
            ConnectionGene.pool[self.connection] = self.innovation_number

    def copy(self):
        """Make a copy of this gene.

        Returns: the copy of this gene.
        """
        copy = ConnectionGene()
        copy.connection = self.connection.copy()
        copy.innovation_number = self.innovation_number

        return copy

    def __str__(self):
        return 'Connection_Gene_%d(%s)' % (self.innovation_number, self.connection)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.connection == other.connection and self.innovation_number == other.innovation_number

    def __hash__(self):
        return self.innovation_number


class Genome:
    """Represents a creature's genome (a set of genes)."""

    def __init__(self):
        self.node_genes = []
        self.connection_genes = set()

    def copy(self):
        """Make a copy of a genome.

        Returns: The copy of the genome.
        """
        copy = Genome()
        copy.node_genes = [node_gene.copy() for node_gene in self.node_genes]
        copy.connection_genes = set(connection_gene.copy() for connection_gene in self.connection_genes)

        return copy

    def add_gene(self, gene):
        """Add a gene to the genome.

        The innovation number of the gene is also set in this method.

        Arguments:
            gene: The gene to be added to the genome.
        """
        if isinstance(gene, NodeGene):
            self.node_genes.append(gene)
        else:
            self.connection_genes.add(gene)

    def add_genes(self, genes):
        """Add a list of genes to the genome.

        Arguments:
            genes: a list of Gene objects that are to be added.
        """
        for gene in genes:
            self.add_gene(gene)

    def align_genes(self, other_genotype):
        """Find the aligned, disjoint, and excess genes of two genotypes.

        Arguments:
            other_genotype: the genotype to align with.

        Returns: a 3-tuple where the elements are a set of aligned genes,
                 disjoint genes, and excess genes. The aligned genes element
                 itself is also a tuple which contains the sets of aligned
                 genes for each genotype.
        """
        genes = self.connection_genes
        other_genes = other_genotype.connection_genes
        excess_threshold = min(self.max_innovation_number, other_genotype.max_innovation_number)

        # We assign aligned_genes to the two mirrored intersections so that we get the aligned genes of both genotypes.
        # Only taking one intersection ignores one creature's aligned genes.
        # The first element is the set of aligned genes from this genotype, and the other element is the aligned
        # genes from the other genotype.
        aligned_genes = (other_genes.intersection(genes), genes.intersection(other_genes))
        unaligned_genes = genes.symmetric_difference(other_genes)
        disjoint_genes = set(filter(lambda gene: gene.innovation_number <= excess_threshold, unaligned_genes))
        excess_genes = unaligned_genes.difference(disjoint_genes)

        return aligned_genes, disjoint_genes, excess_genes

    @property
    def max_innovation_number(self):
        """The highest innovation number in the genotype's connection genes."""
        return max([gene.innovation_number for gene in self.connection_genes])

    def __len__(self):
        """Get the length of the genome.

        The length is simply the number of node genes plus the number of
        connection genes.

        Returns: the length of the genome.
        """
        return len(self.node_genes) + len(self.connection_genes)


class Phenotype(Graph):
    """A phenotype, or physical expression, of a genome (genotype)."""

    def __init__(self, genome):
        """Generate the phenotype, or physical expression, of a genome.

        Arguments:
            genome: the genome to express.
        """
        super().__init__()

        for node_gene in genome.node_genes:
            node = node_gene.node
            node.id = genome.node_genes.index(node_gene)

            self.add_node(node)

        for connection_gene in genome.connection_genes:
            self.add_connection(connection_gene.connection)

        self.compile()
