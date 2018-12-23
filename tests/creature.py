import json
import random
import unittest

from neat.creature import Creature


class CreatureUnitTest(unittest.TestCase):
    def test_creature_json(self):
        """Test whether a creature object can be saved to and loaded from JSON.
        """
        creature = Creature(4, 1)
        dump = json.dumps(creature.to_json())
        creature_load = Creature.from_json(json.loads(dump))

        self.assertEqual(creature, creature_load)


if __name__ == '__main__':
    random.seed(42)

    unittest.main()
