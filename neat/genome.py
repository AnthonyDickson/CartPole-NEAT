"""Implements a basic model of genomes (genotypes), and phenotypes of
creatures in NEAT.
"""
import random

from neat.gene import NodeGene, ConnectionGene
from neat.graph import Hidden, Sensor, Output


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
    def enabled_connection_genes(self):
        return list(filter(lambda cg: cg.is_enabled, self.connection_genes))

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

    def align_genes(self, other_genotype, is_dominant):
        """Find the aligned, disjoint, and excess genes of two genotypes.

        Arguments:
            other_genotype: the genotype to align with.
            is_dominant: whether the genes in this genotype are to be
                             considered dominant.

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

        if is_dominant:
            unaligned_genes = genes.difference(other_genes)
        else:
            unaligned_genes = other_genes.difference(genes)

        disjoint_genes = set(
            filter(lambda gene: gene.innovation_number <= excess_threshold,
                   unaligned_genes)
        )
        excess_genes = unaligned_genes.difference(disjoint_genes)

        return list(zip(*aligned_genes)), disjoint_genes, excess_genes

    def crossover(self, other, is_dominant=True):
        """Perform crossover between two genotypes.

        If is_dominant is True, the genes in this genotype are considered
        dominant, and unaligned genes are inherited from this genotype.
        Otherwise, the genes in this genotype are considered recessive, and
        unaligned genes are inherited from the other genotype. Aligned genes
        are unaffected by gene dominance.

        Arguments:
                other: the other genotype to crossover with.
                is_dominant: whether the genes in this genotype are to be
                             considered dominant.

        Returns: a new genotype.
        """
        if random.random() < Genome.p_mate_choose:
            genotype = self._crossover(other, Genome._combine_randomly,
                                       is_dominant)
        else:
            genotype = self._crossover(other, Genome._combine_average,
                                       is_dominant)

        if random.random() < Genome.p_re_enable_connection:
            self._reenable_random_connection()

        return genotype

    def _crossover(self, other, combining_method, is_dominant):
        """Perform crossover between two genotypes by choosing genes randomly
        from each parent.

        The genotype that this method is called on is considered the dominant
        genotype, and genes will be inherited from this genotype when choosing
        between this genotype and the other.

        Arguments:
                other: the other genotype to crossover with.
                combining_method: How to combine the aligned genes. Should be
                                  either _combine_randomly or _combine_average.
                is_dominant: whether the genes in this genotype are to be
                             considered dominant.

        Returns: a new genotype.
        """
        node_genes = self._choose_genes(other, self._align_node_genes,
                                        combining_method, is_dominant)
        connection_genes = set(self._choose_genes(other, self.align_genes,
                                                  combining_method,
                                                  is_dominant))

        genotype = Genome()
        genotype.add_genes(node_genes)
        genotype.add_genes(connection_genes)

        return genotype

    def _choose_genes(self, other, alignment_method, combining_method,
                      is_dominant):
        """Randomly choose node genes from each genotype.

        The aligned genes are each taken from a random genotype and the
        unaligned genes (i.e. disjointed and excess genes) are all inherited.

        Arguments:
                other: the other genotype to crossover with.
                alignment_method: How to align the genes, also decides which
                                  genes to align. Should be either align_genes
                                  or _align_node_genes.
                combining_method: How to combine the aligned genes. Should be
                                  either _combine_randomly or _combine_average.
                is_dominant: whether the genes in this genotype are to be
                             considered dominant.

        Returns: the selection of node genes from both genotypes.
        """
        if alignment_method == self.align_genes:
            averaging_attribute = 'weight'
        else:
            averaging_attribute = 'bias'

        aligned, disjoint, excess = alignment_method(other, is_dominant)

        if combining_method == Genome._combine_randomly:
            genes = combining_method(aligned)
        else:  # combining via averaging
            genes = combining_method(aligned, averaging_attribute)

        genes += disjoint
        genes += excess

        genes = sorted(genes)

        return genes

    def _align_node_genes(self, other, is_dominant):
        """Align the node genes between two genotypes.

        Arguments:
                other: the other genotype to align with.
                is_dominant: whether the genes in this genotype are to be
                             considered dominant.

        Returns: a tuple containing a list of aligned gene pairs, disjoint
                 genes, and excess genes.
        """
        aligned_with_self = set(other.node_genes) \
            .intersection(self.node_genes)
        aligned_with_other = set(self.node_genes) \
            .intersection(other.node_genes)
        aligned_node_genes = zip(aligned_with_self, aligned_with_other)

        if is_dominant:
            unaligned_node_genes = set(self.node_genes) \
                .difference(other.node_genes)
        else:
            unaligned_node_genes = set(other.node_genes) \
                .difference(self.node_genes)

        max_node_id = max(self.node_genes).node.id
        max_other_node_id = max(other.node_genes).node.id

        excess_threshold = min(max_node_id, max_other_node_id)

        disjointed = set(filter(lambda ng: ng.node.id <= excess_threshold,
                                unaligned_node_genes))
        excess = unaligned_node_genes.difference(disjointed)

        return list(aligned_node_genes), list(disjointed), list(excess)

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

    @staticmethod
    def _combine_average(aligned, attribute):
        """Combines list of aligned gene pairs by averaging a common attribute.

        Arguments;
            aligned: the list of aligned gene pairs.
            container: the encapsulating .
            attribute: the attribute to average.

        Returns: a list of averaged genes.
        """
        selection = []

        for gene1, gene2 in aligned:
            gene = gene1.copy()

            avg = 0.5 * (getattr(gene1, attribute) + getattr(gene2, attribute))
            setattr(gene, attribute, avg)

            selection.append(gene)

        return selection

    def _reenable_random_connection(self):
        """Re-enable a previously disabled connection gene."""
        disabled_genes = list(filter(lambda cg: not cg.is_enabled,
                                     self.connection_genes))

        if len(disabled_genes) == 0:
            return

        gene = random.choice(disabled_genes)
        gene.connection.is_enabled = True

    def mutate(self):
        """Mutate a given genotype."""
        self._perturb()

        if random.random() < Genome.p_add_node:
            self._give_extra_brain_cell()
        elif random.random() < Genome.p_add_connection:
            self._build_bridges_not_walls()

    def _perturb(self):
        """Add a small positive or negative number to the weights and biases
        in the connection and node genes.
        """
        for node_gene in self.node_genes:
            if random.random() < Genome.p_perturb:
                node_gene.node.bias += random.gauss(0, Genome.perturb_range)

        for connection_gene in self.connection_genes:
            if connection_gene.is_enabled and \
                    random.random() < Genome.p_perturb:
                connection_gene.connection.weight += \
                    random.gauss(0, Genome.perturb_range)

    def _give_extra_brain_cell(self):
        """Add a new node to the genome via mutation.

        This process chooses a random enabled connection, and splits it into
        two new connections with a new node in the middle.
        """
        enabled_connections = filter(lambda cg: cg.is_enabled,
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
        nodes = [ng.node for ng in self.node_genes]
        target_node, input_node = random.choices(nodes, k=2)

        # Sensor nodes 'reject' incoming connections.
        # No recurrent connections from Output nodes.
        while isinstance(target_node, Sensor) or \
                isinstance(input_node, Output):
            target_node, input_node = random.choices(nodes, k=2)

        self.add_gene(ConnectionGene(target_node.id, input_node.id))

    def __len__(self):
        """Get the length of the genome.

        The length is simply the number of node genes plus the number of
        connection genes.

        Returns: the length of the genome.
        """
        return len(self.node_genes) + len(self.connection_genes)


