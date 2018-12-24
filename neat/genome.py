"""Implements a basic model of genomes (genotypes), and phenotypes of
creatures in NEAT.
"""
import random

from neat.gene import NodeGene, ConnectionGene
from neat.node import Hidden, Sensor, Output


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
    p_add_node = 0.06

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

    def crossover(self, other, dominance):
        """Perform crossover between two genotypes.

        If is_dominant is True, the genes in this genotype are considered
        dominant, and unaligned genes are inherited from this genotype.
        Otherwise, the genes in this genotype are considered recessive, and
        unaligned genes are inherited from the other genotype. Aligned genes
        are unaffected by gene dominance.

        Arguments:
                other: the other genotype to crossover with.
                dominance: the relative composite fitness between the creature
                           who owns this genotype and the creature who owns the
                           other genotype.

        Returns: a new genotype.
        """
        combine_by_average = random.random() < Genome.p_mate_average

        node_genes = Genome._choose(self.node_genes, other.node_genes,
                                    combine_by_average, dominance)
        connection_genes = self._choose(self.connection_genes,
                                        other.connection_genes,
                                        combine_by_average, dominance)

        genotype = Genome()
        genotype.add_genes(node_genes)
        genotype.add_genes(connection_genes)

        if random.random() < Genome.p_re_enable_connection:
            genotype._reenable_random_connection()

        return genotype

    @staticmethod
    def _choose(genes, other_genes, combine_by_average, dominance):
        """Given two lists of genes, combine aligned gene pairs and unaligned
        genes.

        Each aligned gene pair is reduced to a single gene. The disjointed and
        excess genes are inherited from the dominant parent.

        Arguments:
                genes: the first set of genes to choose from.
                other_genes: the other genes to choose from.
                combine_by_average: Combine aligned genes by averaging traits
                                    if True, otherwise combine randomly.
                dominance: the relative composite fitness between the creature
                           who owns this genotype and the creature who owns the
                           other genotype.


        Returns: the selection of node genes from both genotypes.
        """
        aligned, disjoint, excess = Genome.align_genes(genes, other_genes,
                                                       dominance)

        if combine_by_average:
            genes = [gene1.combine_by_average(gene2)
                     for gene1, gene2 in aligned]
        else:
            genes = [gene1 if random.random() < 0.5 else gene2
                     for gene1, gene2 in aligned]

        genes += disjoint
        genes += excess

        genes = sorted(genes)

        return genes

    @staticmethod
    def align_genes(genes, other_genes, dominance):
        """Align two sets of genes.

        If there are any unaligned genes, then they are taken from the dominant
        genotype unless they are equally dominant (i.e. their respective
        creatures have identical fitness), in which case all unaligned genes
        are retained.

        Arguments:
                genes: the first set of genes to align. Typically a genotype's
                       node_genes or connection_genes attribute.
                other_genes: the other genotype to align with.
                dominance: the relative composite fitness between the creature
                           who owns this genotype and the creature who owns the
                           other genotype.

        Returns: a tuple containing a list of aligned gene pairs, disjoint
                 genes, and excess genes.
        """
        genes = set(genes)
        other_genes = set(other_genes)
        excess_threshold = \
            min(max([gene.alignment_key for gene in genes]),
                max([gene.alignment_key for gene in other_genes]))

        aligned_genes = (other_genes.intersection(genes),
                         genes.intersection(other_genes))

        if dominance >= 0:
            unaligned_genes = genes.difference(other_genes)
        else:
            unaligned_genes = other_genes.difference(genes)

        disjointed = \
            set(filter(lambda gene: gene.alignment_key <= excess_threshold,
                       unaligned_genes))
        excess = unaligned_genes.difference(disjointed)

        return list(zip(*aligned_genes)), list(disjointed), list(excess)

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
        new_node = NodeGene(Hidden())
        new_node.node.bias = 0
        new_node.node.id = len(self.node_genes)
        self.add_gene(new_node)

        enabled_connections = list(filter(lambda cg: cg.is_enabled,
                                          self.connection_genes))
        connection_to_split = random.choice(enabled_connections)
        connection_to_split.connection.is_enabled = False

        first_connection = \
            ConnectionGene(connection_to_split.connection.origin_id,
                           new_node.node.id)
        first_connection.connection.weight = 1.0
        self.add_gene(first_connection)

        second_connection = \
            ConnectionGene(new_node.node.id,
                           connection_to_split.connection.target_id)
        second_connection.connection.weight = \
            connection_to_split.connection.weight
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

    def to_json(self):
        """Encode the genotype as JSON.

        Returns: the JSON encoded genotype.
        """
        return dict(
            node_genes=[ng.to_json() for ng in self.node_genes],
            connection_genes=[cg.to_json() for cg in self.connection_genes]
        )

    @staticmethod
    def from_json(config):
        """Load a genome object from JSON.

        Arguments:
            config: the JSON dictionary loaded from file.

        Returns: a genome object.
        """
        genotype = Genome()
        genotype.add_genes([NodeGene.from_json(ng_config) for
                            ng_config in config['node_genes']])
        genotype.add_genes([ConnectionGene.from_json(cg_config) for
                            cg_config in config['connection_genes']])

        return genotype

    def __len__(self):
        """Get the length of the genome.

        The length is simply the number of node genes plus the number of
        connection genes.

        Returns: the length of the genome.
        """
        return len(self.node_genes) + len(self.connection_genes)


