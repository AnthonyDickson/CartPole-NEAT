"""Implements stuff related to species."""

import random
import re

class CodeNameGenerator:
    """Generates a random name based on adjectives and nouns used in,
    and proposed for Ubuntu release code names.

    Code names consist of an adjective and an animal name that starts
    with the same letter as the adjective (i.e. a tautogram).

    Sourced from https://wiki.ubuntu.com/DevelopmentCodeNames
    """

    marker_pattern = re.compile(r"^\[[A-Za-z]\]$")
    key_pattern = re.compile(r"^\[([A-Za-z])\]$")
    comment_pattern = re.compile(r"^#.*")

    def __init__(self, data_path='neat/data/ubuntu/', adjective_file='adjectives.txt', \
            noun_file='nouns.txt'):
        """Create a name generator based on Ubuntu code names.

        Data files should have the following format:
            - Comments start the line with a # symbol.
            - Words should be grouped by the first letter of the word.
            - A marker should be placed at the top of  each group of words,
                and it should contain the corresponding letter captilised and
                surrounded in brackets. For example, all adjectives starting
                with 'A' or 'a' should be grouped togetherand come after the
                marker '[A]'.

        Arguments:
            data_path: where the data files containing the adjective and animal names are located.
            adjective_files: the name of the file that contains the adjectives.
            noun_file: the name of the file that contains the animal names.
        """
        filepath = data_path + adjective_file

        with open(filepath, 'r') as file:
            self.adjectives = CodeNameGenerator.make_dict(file)

        filepath = data_path + noun_file

        with open(filepath, 'r') as file:
            self.nouns = CodeNameGenerator.make_dict(file)

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
            line = CodeNameGenerator.process(line)

            if re.match(CodeNameGenerator.comment_pattern, line):
                continue

            if re.match(CodeNameGenerator.marker_pattern, line):
                result = re.search(CodeNameGenerator.key_pattern, line)
                key = result.group(1)
                word_dict[key] = []
                curr_key = key
            else:
                word_dict[curr_key].append(line)

        return word_dict

    @staticmethod
    def process(line):
        """Process a line and make it ready for use.

        Returns: the line, stripped of trailing whitespace and all words capitalised.
        """
        line = line.strip()
        line = CodeNameGenerator.capitalise(line)

        return line

    @staticmethod
    def capitalise(string):
        """Capitalise the words in the string.

        Returns: the string, with all words capitalised.
        """
        return ' '.join(map(CodeNameGenerator.capitalise_hyphened, string.split()))

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

    @staticmethod
    @property
    def next_id():
        """Get the next species id.

        Returns: an integer representing the next species id.
        """
        Species.species_count += 1

        return Species.species_count

    @staticmethod
    @property
    def next_name():
        """Get the next species name.

        Returns: a name.
        """
        if Species.name_generator is None:
            Species.name_generator = CodeNameGenerator()

        return Species.name_generator.next()

    def __init__(self, name=''):
        """Create a new species.

        Arguments:
            name: The name of the species.
        """
        self.id = Species.next_id
        self.name = name if name != '' else Species.next_name
        self.members = set()
        self.representative = None

    def __str__(self):
        return '%s (Species_%d)' % (self.name, self.id)

    def __hash__(self):
        return self.id

if __name__ == '__main__':
    print(CodeNameGenerator().next())