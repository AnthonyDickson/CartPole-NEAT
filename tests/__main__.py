import sys
import unittest

from tests import genome
from tests import graph, species


# noinspection PyTypeChecker
def main():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromModule(graph))
    suite.addTests(loader.loadTestsFromModule(genome))
    suite.addTests(loader.loadTestsFromModule(species))

    runner = unittest.TextTestRunner()
    result = runner.run(suite)

    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
