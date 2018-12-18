"""Unit tests for the species module."""

import os
import random
import re
import unittest

from neat.species import NameGenerator, UbuntuCodeName

class NameGenerationUnitTest(unittest.TestCase):
    """Test cases for the name generation code in the species module."""

    # Tests are designed to be run from the repo's root directory.
    data_path = os.getcwd() + '/neat/data/ubuntu/'

    def test_string_preprocessing(self):
        """Check if lines are correctly processed.

        Success if all trailing white space is removed, and all words are capitalised.
        """
        tests = [
            (' aa aa \n', 'Aa Aa'),
            ('oompa-loompa', 'Oompa-Loompa')
        ]

        for test_string, expected in tests:
            actual = NameGenerator.process(test_string)
            self.assertEqual(actual, expected, \
                'Expected %s, but got %s.' % (expected, actual))

    def test_loads_from_file(self):
        """Attempt to intialise a UbuntuCodeName name generator.

        Success if no errors raised and things are loaded correctly.
        """
        name_gen = UbuntuCodeName(NameGenerationUnitTest.data_path)

        self.assertGreater(len(name_gen.adjectives), 0)
        self.assertGreater(len(name_gen.nouns), 0)

    def test_generates_names(self):
        """Briefly test name generation.

        Success if generated names a in correct format and can generate n names consecutively.
        """
        name_gen = UbuntuCodeName(NameGenerationUnitTest.data_path)

        capitalised_words = re.compile(r"^([A-Z][a-z]+('s)?[\s-])+([A-Z][a-zíñó]+)$")

        for _ in range(10000):
            name = name_gen.next()

            self.assertGreater(len(name), 0)
            self.assertFalse(re.match(capitalised_words, name) is None, \
                'Expected capitalised words, got '\
                '\'%s\'' % name)

if __name__ == '__main__':
    random.seed(42)

    unittest.main()
