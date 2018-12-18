import sys
import unittest

from neat.tests import graph, genome, species

def main():
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromModule(graph))
    suite.addTests(loader.loadTestsFromModule(genome))
    suite.addTests(loader.loadTestsFromModule(species))

    runner = unittest.TextTestRunner()
    result = runner.run(suite)

    sys.exit(0 if result.wasSuccessful() else 1)

if __name__ == '__main__':
    main()