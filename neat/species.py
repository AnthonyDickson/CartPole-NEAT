"""Implements stuff related to species."""

import random
import re


class NameGenerator:
    """Generates a random name based on adjectives and nouns used in,
    similar to Ubuntu release code names.

    Names consist of an adjective and an animal name that starts
    with the same letter as the adjective (i.e. a tautogram).
    """

    marker_pattern = re.compile(r"^\[[A-Za-z]\]$")
    key_pattern = re.compile(r"^\[([A-Za-z])\]$")
    comment_pattern = re.compile(r"^#.*")

    def __init__(self, data_path='neat/data/',
                 adjective_file='adjectives.txt',
                 noun_file='nouns.txt'):
        """Create a name generator based on Ubuntu code names.

        Data files should have the following format:
            - Comments start the line with a # symbol.
            - Words should be grouped by the first letter of the word.
            - A marker should be placed at the top of  each group of words,
                and it should contain the corresponding letter capitalised and
                surrounded in brackets. For example, all adjectives starting
                with 'A' or 'a' should be grouped together and come after the
                marker '[A]'.

        Arguments:
            data_path: where the data files containing the adjective and animal
                       names are located.
            adjective_file: the name of the file that contains the adjectives.
            noun_file: the name of the file that contains the animal names.
        """
        filepath = data_path + adjective_file

        with open(filepath, 'r') as file:
            self.adjectives = NameGenerator.make_dict(file)

        filepath = data_path + noun_file

        with open(filepath, 'r') as file:
            self.nouns = NameGenerator.make_dict(file)

    @staticmethod
    def make_dict(file):
        """Read a data file and generate a dictionary from it.

        Arguments:
            file: the opened file that contains the data.

        Returns: a dictionary of words, indexed by starting letter.
        """
        word_dict = {}
        curr_key = ''

        for line in file:
            line = NameGenerator.process(line)

            if re.match(NameGenerator.comment_pattern, line):
                continue

            if re.match(NameGenerator.marker_pattern, line):
                result = re.search(NameGenerator.key_pattern, line)
                key = result.group(1)
                word_dict[key] = []
                curr_key = key
            else:
                word_dict[curr_key].append(line)

        return word_dict

    @staticmethod
    def process(line):
        """Process a line and make it ready for use.

        Returns: the line, stripped of trailing whitespace and all words
        capitalised.
        """
        line = line.strip()
        line = NameGenerator.capitalise(line)

        return line

    @staticmethod
    def capitalise(string):
        """Capitalise the words in the string.

        Returns: the string, with all words capitalised.
        """
        return ' '.join(map(NameGenerator.capitalise_hyphened,
                            string.split()))

    @staticmethod
    def capitalise_hyphened(string):
        """Capitalise the string, including words separated by hyphens.

        Returns: the string, with all words capitalised.
        """
        return '-'.join(map(str.capitalize, string.split('-')))

    def next(self):
        """Gets the next name.

        Returns: the next name.
        """
        key = random.choice(list(self.adjectives.keys()))
        adjective = random.choice(self.adjectives[key])
        noun = random.choice(self.nouns[key])

        return ' '.join([adjective, noun])


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

    @staticmethod
    def to_ordinal(n):
        """Get the suffixed number.

        Returns: a string with then number  and the appropriate suffix.

        >>> Species.to_ordinal(1)
        '1st'
        >>> Species.to_ordinal(2)
        '2nd'
        >>> Species.to_ordinal(3)
        '3rd'
        >>> Species.to_ordinal(4)
        '4th'
        >>> Species.to_ordinal(11)
        '11th'
        >>> Species.to_ordinal(12)
        '12th'
        >>> Species.to_ordinal(13)
        '13th'
        """
        if 11 <= (n % 100) <= 13:
            return '%dth' % n

        last_digit = n % 10

        if last_digit == 1:
            return '%dst' % n
        elif last_digit == 2:
            return '%dnd' % n
        elif last_digit == 3:
            return '%drd' % n
        else:
            return '%dth' % n

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
        self.champion = None
        self.is_extinct = False
        self.age = 0  # How many generations the species has survived.
        # Number of creatures in the species, past and present.
        self.total_num_members = 0

    @property
    def mean_fitness(self):
        """The mean fitness of the entire species."""
        return sum([creature.fitness for creature in self.members]) / \
               len(self.members)

    def add(self, creature):
        """Add a creature to the species.

        Arguments:
            creature: the creature to be added to the species.
        """
        self.members.append(creature)
        self.total_num_members += 1
        creature.name_suffix = 'the %s' % \
                               Species.to_ordinal(self.total_num_members)
        creature.species = self

    def assign_members(self, members):
        """Assign all the members of this species.

        Arguments:
            members: the new members of the species.
        """
        self.members = []

        for creature in sorted(members):
            self.add(creature)

        self.representative = random.choice(members)

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
        # The champion is the last in the list because the creatures are
        # ranked in order of increasing fitness.
        self.champion = self.members[-1]
        num_to_kill = int(how_many * len(self.members))
        survivor_list = list(self.members[num_to_kill:])
        self.assign_members(survivor_list)

        return survivor_list

    def next_generation(self, generation_champ, population):
        """Get the species' next generation of creatures.

        Arguments:
            generation_champ: the best creature for the whole generation, who's
                              just an all-round champ.
            population: the entire population of creatures, including the champ
                        and the rest of the chumps in the generation.

        Returns: a list of new creatures generated via crossover. Up to the
                 allotted number of offspring will be created.
        """
        offspring = []
        pool = self.members

        if self.allotted_offspring_quota > 1:
            offspring.append(self.champion.copy())

        if len(offspring) < self.allotted_offspring_quota:
            offspring.append(self.champion.mate(generation_champ))

        while len(offspring) < self.allotted_offspring_quota:
            parent1 = random.choice(pool)

            if random.random() < Species.p_interspecies_mating:
                parent2 = random.choice(population)
            else:
                parent2 = random.choice(pool)

            # The parent who 'initiates' crossover (calls crossover) is
            # considered the dominant parent, so matter order matters.
            if parent1.fitness >= parent2.fitness:
                offspring.append(parent1.mate(parent2))
            else:
                offspring.append(parent2.mate(parent1))

        if len(offspring) == 0:
            self.is_extinct = True  # R.I.P.
        else:
            self.assign_members(offspring)
            self.age += 1

        return offspring

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


if __name__ == '__main__':
    print(NameGenerator().next())
