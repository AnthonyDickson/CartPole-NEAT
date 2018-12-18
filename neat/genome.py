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
        return 'Gene_Node(%s)' % self.node

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
            self.innovation_number = ConnectionGene.pool[self]
        except KeyError:
            self.innovation_number = len(ConnectionGene.pool)
            ConnectionGene.pool[self] = self.innovation_number

    def copy(self):
        """Make a copy of this gene.

        Returns: the copy of this gene.
        """
        copy = ConnectionGene()
        copy.connection = self.connection.copy()
        copy.innovation_number = self.innovation_number

        return copy

    def __str__(self):
        return 'Gene_Connection(%s)' % self.connection

    def __eq__(self, other):
        return self.connection == other.connection

    def __hash__(self):
        return hash(self.connection)

class Genome:
    """Represents a creature's genome (a set of genes)."""

    def __init__(self):
        self.node_genes = []
        self.connection_genes = []

    def copy(self):
        """Make a copy of a genome.

        Returns: The copy of the genome.
        """
        copy = Genome()
        copy.node_genes = [node_gene.copy() for node_gene in self.node_genes]
        copy.connection_genes = [connection_gene.copy() \
            for connection_gene in self.connection_genes]

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
            self.connection_genes.append(gene)

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

        Returns: a 3-tuple where the elements are a list of aligned genes,
                 disjoint genes, and excess genes. The aligned genes element
                 itself is also a tuple which contains the pairs of aligned
                 genes.
        """
        genes = self.connection_genes
        other_genes = other_genotype.connection_genes

        min_gene_length = min(len(genes), len(other_genes))

        max_innovation_number = max([gene.innovation_number for gene in genes])
        other_max_innovation_number = max([gene.innovation_number for gene in other_genes])
        excess_threshold = min(max_innovation_number, other_max_innovation_number)

        aligned_genes = []
        disjoint_genes = []
        excess_genes = []

        for i in range(min_gene_length):
            innovation_number = genes[i].innovation_number
            other_innovation_number = other_genes[i].innovation_number

            if innovation_number == other_innovation_number:
                aligned_genes.append((genes[i], other_genes[i]))
            else:
                if innovation_number <= excess_threshold:
                    disjoint_genes.append(genes[i])
                else:
                    excess_genes.append(genes[i])

                if other_innovation_number <= excess_threshold:
                    disjoint_genes.append(other_genes[i])
                else:
                    excess_genes.append(other_genes[i])

        return aligned_genes, disjoint_genes, excess_genes

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
