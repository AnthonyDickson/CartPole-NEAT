import sys
import unittest

from tests import genome, graph, species, neat_main


# noinspection PyTypeChecker
def main():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromModule(graph))
    suite.addTests(loader.loadTestsFromModule(genome))
    suite.addTests(loader.loadTestsFromModule(species))
    suite.addTests(loader.loadTestsFromModule(neat_main))

    runner = unittest.TextTestRunner()
    result = runner.run(suite)

    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
