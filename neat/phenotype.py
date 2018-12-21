from neat.graph import Graph


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

        for connection_gene in genome.enabled_connection_genes:
            self.add_connection(connection_gene.connection)

        self.compile()
