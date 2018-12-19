"""Unit tests for the main module."""

import os
import random
import sys
import unittest

import gym

from neat.main import NeatAlgorithm


# noinspection PyMethodMayBeStatic
class NeatAlgorithmUnitTest(unittest.TestCase):
    """Test cases for the neat algorithm in the main module."""

    def test_runs_without_setting_fire_to_the_server_room(self):
        """Test whether the main training loop can be run end to end without blowing up."""
        f = None

        try:
            f = open(os.devnull, 'w')
            sys.stdout = f

            env = gym.make('CartPole-v0')
            neat = NeatAlgorithm(env)

            neat.train()
        except Exception as error:
            raise error
        finally:
            sys.stdout = sys.__stdout__
            f.close()


if __name__ == '__main__':
    random.seed(42)

    unittest.main()
