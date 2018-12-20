"""Implements a basic model of genes, genomes (genotypes), and phenotypes of
creatures in NEAT.
"""
import random

from neat.graph import Graph, Connection, Hidden


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
        copy.innovation_number = self.innovation_number

        return copy

    def __str__(self):
        return 'Connection_Gene_%d(%s)' % (self.innovation_number,
                                           self.connection)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.connection == other.connection and \
               self.innovation_number == other.innovation_number

    def __hash__(self):
        return self.innovation_number


class Genome:
    """Represents a creature's genome (a set of genes)."""

    # The below parameters are for controlling crossover.

    # The probability that the genes of the next offspring will be chosen
    # randomly from each parent.
    p_mate_choose = 0.6

    # The probability that the genes of the next offspring will be chosen by
    # averaging the weights between parents.
    p_mate_average = 1 - p_mate_choose

    # The below parameters are for controlling mutation.

    # The probability to add a node gene.
    p_add_node = 0.012

    # The probability to add a connection gene.
    p_add_connection = 0.06

    # The probability that the next connection gene will be a recurrent one.
    p_recurrent_connection = 0.2

    # The probability that a disabled connection gene is re-enabled.
    p_re_enable_connection = 0.03

    # The probability to perturb a weight or bias.
    p_perturb = 0.1

    # The range of perturbations that can occur to a weight or bias.
    perturb_range = 1.0

    def __init__(self):
        self.node_genes = []
        self.connection_genes = set()
        self.mutation_method = ''

    def copy(self):
        """Make a copy of a genome.

        Returns: The copy of the genome.
        """
        copy = Genome()
        copy.node_genes = [node_gene.copy() for node_gene in self.node_genes]
        copy.connection_genes = set(connection_gene.copy() for connection_gene
                                    in self.connection_genes)

        return copy

    @property
    def all_genes(self):
        """All of the genotype's genes."""
        return list(self.node_genes) + list(self.connection_genes)

    @property
    def max_innovation_number(self):
        """The highest innovation number in the genotype's connection genes."""
        return max([gene.innovation_number for gene in self.connection_genes])

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
                 itself is a list containing aligned gene pairs.
        """
        genes = self.connection_genes
        other_genes = other_genotype.connection_genes
        excess_threshold = min(self.max_innovation_number,
                               other_genotype.max_innovation_number)

        # We assign aligned_genes to the two mirrored intersections so that
        # we get the aligned genes of both genotypes. Only taking one
        # intersection ignores one creature's aligned genes. The first element
        # is the set of aligned genes from this genotype, and the other element
        # is the aligned genes from the other genotype.
        aligned_genes = (other_genes.intersection(genes),
                         genes.intersection(other_genes))
        unaligned_genes = genes.symmetric_difference(other_genes)
        disjoint_genes = set(
            filter(lambda gene: gene.innovation_number <= excess_threshold,
                   unaligned_genes)
        )
        excess_genes = unaligned_genes.difference(disjoint_genes)

        return list(zip(*aligned_genes)), disjoint_genes, excess_genes

    def crossover(self, other):
        """Perform crossover between two genotypes.

        The genotype that this method is called on is considered the dominant
        genotype, and genes will be inherited from this genotype when choosing
        between this genotype and the other.

        Arguments:
                other: the other genotype to crossover with.

        Returns: a new genotype.
        """
        if random.random() < Genome.p_mate_choose:
            genotype = self._crossover_choose(other)
        else:
            genotype = self._crossover_average(other)

        for gene in self.connection_genes:
            if not gene.connection.is_enabled and \
                    random.random() < Genome.p_re_enable_connection:
                gene.connection.is_enabled = True

        return genotype

    def _crossover_choose(self, other):
        """Perform crossover between two genotypes by choosing genes randomly
        from each parent.

        The genotype that this method is called on is considered the dominant
        genotype, and genes will be inherited from this genotype when choosing
        between this genotype and the other.

        Arguments:
                other: the other genotype to crossover with.

        Returns: a new genotype.
        """
        node_genes = self._choose_node_genes(other)
        connection_genes = self._choose_connection_genes(other)

        genotype = Genome()
        genotype.add_genes(node_genes)
        genotype.add_genes(connection_genes)

        return genotype

    def _choose_node_genes(self, other):
        """Randomly choose node genes from each genotype.

        The aligned genes are each taken from a random genotype and the
        unaligned genes (i.e. disjointed and excess genes) are all inherited.

        Arguments:
                other: the other genotype to crossover with.

        Returns: the selection of node genes from both genotypes.
        """
        aligned_node_genes, unaligned_node_genes = \
            self._align_node_genes(other)

        node_genes = Genome._combine_randomly(aligned_node_genes)
        node_genes += unaligned_node_genes
        node_genes = sorted(node_genes, key=lambda ng: ng.node.id)

        return node_genes

    def _align_node_genes(self, other):
        """Align the node genes between two genotypes.

        Arguments:
                other: the other genotype to align with.

        Returns: a tuple containing a list of aligned gene pairs and a list of
                 unaligned node genes (both disjoint and excess).
        """
        aligned_with_self = set(other.node_genes) \
            .intersection(self.node_genes)
        aligned_with_other = set(self.node_genes) \
            .intersection(other.node_genes)
        aligned_node_genes = zip(aligned_with_self, aligned_with_other)

        unaligned_node_genes = set(self.node_genes) \
            .symmetric_difference(other.node_genes)

        return list(aligned_node_genes), list(unaligned_node_genes)

    def _choose_connection_genes(self, other):
        """Randomly choose connection genes from each genotype.

        The aligned genes are each taken from a random genotype and the
        unaligned genes (i.e. disjointed and excess genes) are all inherited.

        Arguments:
                other: the other genotype to crossover with.

        Returns: the selection of connection genes from both genotypes.
        """
        aligned, disjointed, excess = self.align_genes(other)

        connection_genes = set(Genome._combine_randomly(aligned))
        connection_genes.update(disjointed)
        connection_genes.update(excess)

        connection_genes = sorted(connection_genes,
                                  key=lambda cg: cg.innovation_number)
        return connection_genes

    @staticmethod
    def _combine_randomly(aligned):
        """Combines list of aligned gene pairs by randomly selecting one gene
        from each pair.

        More specifically, this method reduces a list of tuples (where each
        tuple represents an aligned gene pair) into a list of genes by
        randomly choosing and copying a single gene from each pair.

        Arguments;
            aligned: the list of aligned gene pairs.

        Returns: a list of randomly selected genes.
        """
        selection = []

        for gene1, gene2 in aligned:
            if random.random() < 0.5:
                selection.append(gene1.copy())
            else:
                selection.append(gene2.copy())

        return selection

    def _crossover_average(self, other):
        """Perform crossover between two genotypes by averaging weights and
        biases in aligned genes.

        The genotype that this method is called on is considered the dominant
        genotype, and genes will be inherited from this genotype when choosing
        between this genotype and the other.

        Arguments:
                other: the other genotype to crossover with.

        Returns: a new genotype.
        """
        node_genes = []

        aligned_node_genes, unaligned_node_genes = \
            self._align_node_genes(other)

        for node_gene, other_node_gene in aligned_node_genes:
            gene = node_gene.copy()
            gene.node.bias = 0.5 * (gene.node.bias +
                                    other_node_gene.node.bias)
            node_genes.append(gene)

        node_genes += unaligned_node_genes
        node_genes = sorted(node_genes, key=lambda ng: ng.node.id)

        connection_genes = []
        aligned, disjoint, excess = self.align_genes(other)

        for gene1, gene2 in aligned:
            gene = gene1.copy()
            gene.connection.weight = 0.5 * (gene1.connection.weight +
                                            gene2.connection.weight)
            connection_genes.append(gene)

        connection_genes += disjoint
        connection_genes += excess

        genotype = Genome()
        genotype.add_genes(node_genes)
        genotype.add_genes(connection_genes)

        return genotype

    def mutate(self):
        """Mutate a given genotype."""
        self._perturb()

        if random.random() < Genome.p_add_node:
            self._give_extra_brain_cell()
            self.mutation_method = 'node'

        if random.random() < Genome.p_add_connection:
            self._build_bridges_not_walls()
            self.mutation_method = 'connection'

    def _perturb(self):
        """Add a small positive or negative number to the weights and biases
        in the connection and node genes.
        """
        for node_gene in self.node_genes:
            if random.random() < Genome.p_perturb:
                node_gene.node.bias += random.gauss(0, Genome.perturb_range)

        for connection_gene in self.connection_genes:
            if random.random() < Genome.p_perturb:
                connection_gene.connection.weight += \
                    random.gauss(0, Genome.perturb_range)

    def _give_extra_brain_cell(self):
        """Add a new node to the genome via mutation.

        This process chooses a random enabled connection, and splits it into
        two new connections with a new node in the middle.
        """
        enabled_connections = filter(lambda cg: cg.connection.is_enabled,
                                     self.connection_genes)
        connection_to_split = random.choice(list(enabled_connections))
        connection_to_split.connection.is_enabled = False

        new_node = NodeGene(Hidden())
        # The new node's bias is set to zero to avoid destabilising the
        # network.
        new_node.node.bias = 0
        new_node.node.id = len(self.node_genes)
        self.add_gene(new_node)

        first_connection = \
            ConnectionGene(connection_to_split.connection.origin_id,
                           new_node.node.id)
        # The first connection weight is set to zero to avoid destabilising
        # the network.
        first_connection.connection.weight = 1.0
        first_connection.connection.is_recurrent = \
            connection_to_split.connection.is_recurrent
        second_connection = \
            ConnectionGene(new_node.node.id,
                           connection_to_split.connection.target_id)
        # The second connection's weight is set to the old connection's weight
        # to avoid destabilising the network.
        second_connection.connection.weight = \
            connection_to_split.connection.weight
        second_connection.connection.is_recurrent = \
            connection_to_split.connection.is_recurrent

        self.add_gene(first_connection)
        self.add_gene(second_connection)

    def _build_bridges_not_walls(self):
        """Add a new connection to the genome via mutation."""
        node_ids = [ng.node.id for ng in self.node_genes]
        node_ids = random.choices(node_ids, k=2)

        node_ids = sorted(node_ids)

        if random.random() < Genome.p_recurrent_connection:
            node_ids = list(reversed(node_ids))

        input_node_id, output_node_id = node_ids
        self.add_gene(ConnectionGene(output_node_id, input_node_id))

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
