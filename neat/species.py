"""Implements stuff related to species."""

import random

from neat.creature import Creature
from neat.name_generation import to_ordinal, NameGenerator


class Species:
    """Represents a species."""
    species_count = 0
    name_generator = None

    # The distance threshold used when deciding if two creatures should
    # belong in the same species or not.
    compatibility_threshold = 3.0

    # The probability that a member of this species will mate with a creature
    # from another species.
    p_interspecies_mating = 0.025

    @staticmethod
    def next_id():
        """Get the next species id.

        Returns: an integer representing the next species id.
        """
        Species.species_count += 1

        return Species.species_count

    @staticmethod
    def next_name():
        """Get the next species name.

        Returns: a name.
        """
        if Species.name_generator is None:
            Species.name_generator = NameGenerator()

        return Species.name_generator.next()

    def __init__(self, name=''):
        """Create a new species.

        Arguments:
            name: The name of the species.
        """
        self.id = Species.next_id()
        self.name = name if name != '' else Species.next_name()
        self.members = list()
        self.representative = None
        self.allotted_offspring_quota = 0
        self.is_extinct = False
        self.age = 0  # How many generations the species has survived.
        # Number of creatures in the species, past and present.
        self.total_num_members = 0

    def copy(self):
        """Make a copy of a species.

        Returns: a copy of the species.
        """
        copy = Species(self.name)
        Species.species_count -= 1
        copy.id = self.id
        copy.members = [member.copy() for member in self.members]
        copy.representative = \
            None if self.representative is None else \
                copy.members[self.members.index(self.representative)]
        copy.allotted_offspring_quota = self.allotted_offspring_quota
        copy.is_extinct = self.is_extinct
        copy.age = self.age
        copy.total_num_members = self.total_num_members

        return copy

    @property
    def mean_fitness(self):
        """The mean fitness of the entire species."""
        return sum([creature.fitness for creature in self.members]) / \
               len(self.members)

    @property
    def champion(self):
        self.members = sorted(self.members)

        return self.members[-1]

    def add(self, creature):
        """Add a creature to the species.

        Arguments:
            creature: the creature to be added to the species.
        """
        self.members.append(creature)
        self.total_num_members += 1
        creature.name_suffix = 'the %s' % to_ordinal(self.total_num_members)
        creature.species = self

    def assign_members(self, members):
        """Assign all the members of this species.

        Arguments:
            members: the new members of the species.
        """
        self.members = []

        for creature in sorted(members):
            self.add(creature)

        self.update_representative()

    def update_representative(self):
        """Choose the next representative."""
        if len(self) == 0:
            self.is_extinct = True
            self.representative = None
        elif self.representative not in self.members:
            self.representative = random.choice(self.members)

    def cull_the_weak(self, how_many):
        """Cull the Weak
        Increase damage against Slowed or Chilled enemies by 20%.

        "I'll show you the same mercy you showed my helpless family."
        —Tyla Shrikewing

        Unlocked at level 20

        Arguments:
            how_many: the ratio of creatures to kill off.

        Returns: the survivors.
        """
        num_to_kill = int(how_many * len(self.members))
        survivor_list = list(self.members[num_to_kill:])
        self.assign_members(survivor_list)

        return survivor_list

    def next_generation(self, generation_champ, population):
        """Get the species' next generation of creatures.

        Arguments:
            generation_champ: the best creature for the whole generation, who's
                              just an all-round champ.
            population: the entire creatures of creatures, including the champ
                        and the rest of the chumps in the generation.

        Returns: a list of new creatures generated via crossover. Up to the
                 allotted number of offspring will be created.
        """
        if self.allotted_offspring_quota == 0:
            self.is_extinct = True  # R.I.P.

            return []

        offspring = []
        pool = self.members
        self.members = sorted(self.members)

        if len(offspring) < self.allotted_offspring_quota:
            offspring.append(self.champion.copy())

        if len(offspring) < self.allotted_offspring_quota:
            offspring.append(self.champion.mate(generation_champ))

        while len(offspring) < self.allotted_offspring_quota:
            parent1 = random.choice(pool)

            if random.random() < Species.p_interspecies_mating:
                parent2 = random.choice(population)
            else:
                parent2 = random.choice(pool)

            offspring.append(parent1.mate(parent2))

        self.assign_members(offspring)
        self.age += 1

        return offspring

    def to_json(self):
        """Encode the gene as JSON.

        Returns: the JSON encoded genes.
        """
        return dict(
            age=self.age,
            id=self.id,
            members=[c.to_json() for c in self.members],
            name=self.name,
            representative=self.members.index(self.representative),
            allotted_offspring_quota=self.allotted_offspring_quota
        )

    @staticmethod
    def from_json(config):
        """Load a gene object from JSON.

        Arguments:
            config: the JSON dictionary loaded from file.

        Returns: a gene object.
        """
        species = Species()
        Species.species_count -= 1

        species.name = config['name']
        species.age = config['age']
        species.id = config['id']
        species.members = [Creature.from_json(c_config)
                           for c_config in config['members']]
        species.representative = species.members[config['representative']]
        species.allotted_offspring_quota = config['allotted_offspring_quota']

        for creature in species.members:
            creature.species = species

        return species

    def __str__(self):
        return '%s' % self.name

    def __repr__(self):
        return '%s (Species_%d)' % (self.name, self.id)

    def __hash__(self):
        return self.id

    def __len__(self):
        """Get the 'length', or size of the species.

        Returns: the number of member creatures in the species.
        """
        return len(self.members)

