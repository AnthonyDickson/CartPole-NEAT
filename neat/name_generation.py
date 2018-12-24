import random
import re


def to_ordinal(n):
    """Get the suffixed number.

    Returns: a string with then number  and the appropriate suffix.

    >>> to_ordinal(1)
    '1st'
    >>> to_ordinal(2)
    '2nd'
    >>> to_ordinal(3)
    '3rd'
    >>> to_ordinal(4)
    '4th'
    >>> to_ordinal(11)
    '11th'
    >>> to_ordinal(12)
    '12th'
    >>> to_ordinal(13)
    '13th'
    """
    if n < 1:
        raise ValueError("n must be at least 1.")

    last_digit = n % 10

    if 11 <= (n % 100) <= 13:
        fmt = '%dth'
    elif last_digit == 1:
        fmt = '%dst'
    elif last_digit == 2:
        fmt = '%dnd'
    elif last_digit == 3:
        fmt = '%drd'
    else:
        fmt = '%dth'

    return fmt % n


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
