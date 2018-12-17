"""Implements a creature that would exist in the NEAT algorithm."""
import numpy as np

from neat.genome import Genome, ConnectionGene, NodeGene, Phenotype
from neat.graph import Sensor, Output

class Creature:
    """A creature that would exist in the NEAT algorithm."""

    def __init__(self, n_inputs=None, n_outputs=None):
        """Create a creature.

        If both n_inputs and n_outputs are set, the creature initially has a fully
        connected neural network with n_inputs input nodes and n_outputs
        output nodes.
        Typically you want to set these parameters when you are first creating
        the seed creature for a population. If you want to create a copy be
        sure to use the copy() method.

        Arguments:
            n_inputs: How many inputs to expect.
            n_outputs: How many outputs are expected - also how many actions the creature can take.
        """
        # This option is here to support copying.
        if n_inputs is None or n_outputs is None:
            self.genotype = None
            self.phenotype = None

            return

        genome = Genome()

        sensors = [NodeGene(Sensor()) for _ in range(n_inputs)]
        outputs = [NodeGene(Output()) for _ in range(n_outputs)]

        connections = []

        for input_id in range(n_inputs):
            for output_id in range(n_inputs, n_inputs + n_outputs):
                connections.append(ConnectionGene(output_id, input_id))

        genome.add_genes(sensors)
        genome.add_genes(outputs)
        genome.add_genes(connections)

        self.genotype = genome
        self.phenotype = Phenotype(genome)

    def copy(self):
        """Make a copy of a creature.

        Returns: the copy of the creature.
        """
        copy = Creature()
        copy.genotype = self.genotype.copy()
        copy.phenotype = self.phenotype.copy()

        return copy

    def get_action(self, x):
        """Get the creature's action for the given input.

        Arguments:
            x: the input, typically an observation of the agent's environment.

        Returns: an integer representing the action the creature will take.
        """
        return np.argmax(self.phenotype.compute(x))
