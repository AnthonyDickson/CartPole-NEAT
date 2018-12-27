"""Unit tests for the main module."""
import json
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
        """Test whether the main training loop can be run end to end without
        blowing up.
        """
        f = None

        try:
            f = open(os.devnull, 'w')
            sys.stdout = f
            sys.stderr = f

            env = gym.make('CartPole-v0')
            neat = NeatAlgorithm(env)

            neat.train(n_episodes=5, debug_mode=True)
        except Exception as error:
            raise error
        finally:
            sys.stderr = sys.__stderr__
            sys.stdout = sys.__stdout__
            f.close()

    def test_json(self):
        """Test if an instance of the NEAT algorithm can be saved to and loaded
        from JSON.
        """
        f = None

        try:
            f = open(os.devnull, 'w')
            sys.stdout = f
            sys.stderr = f

            env = gym.make('CartPole-v0')
            neat = NeatAlgorithm(env)

            neat.train(n_episodes=5, debug_mode=True)

            dump = json.dumps(neat.to_json())
            neat_load = NeatAlgorithm.from_json(json.loads(dump))

            self.assertEqual(neat.run_id, neat_load.run_id)
            self.assertEqual(len(neat.population), len(neat_load.population))
            self.assertEqual(len(neat.species), len(neat_load.species))
            self.assertEqual(neat.env.unwrapped.spec.id,
                             neat_load.env.unwrapped.spec.id)

            # Test that we can run the training loop again.
            neat_load.train(n_episodes=5, debug_mode=True)
        except Exception as error:
            raise error
        finally:
            sys.stderr = sys.__stderr__
            sys.stdout = sys.__stdout__
            f.close()


if __name__ == '__main__':
    random.seed(42)

    unittest.main()
