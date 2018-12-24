"""Implements the NEAT algorithm."""
from time import time

import gym
import numpy as np
from gym import wrappers

from neat.creature import Creature
from neat.genome import Genome
from neat.species import Species


class NeatAlgorithm:
    """An implementation of the NEAT algorithm based off the original paper."""

    # Percentage of each species that is allowed to reproduce.
    survival_threshold = 0.3

    def __init__(self, env, n_pops=150):
        self.env = env
        self.n_pops = n_pops
        self.population = self.init_population(env.observation_space.shape[0],
                                               env.action_space.n)
        self.species = set()

    def init_population(self, n_inputs, n_outputs):
        """Create a population of n individuals.

        Each individual is initially has a fully connected neural network with
        n_inputs neurons in the input layer and n_outputs neurons in the
        output layer.

        Arguments:
            n_inputs: How many inputs to expect. An observation in CartPole
                      has four dimensions, so in this case n_inputs would be
                      four.
            n_outputs: How many outputs to expect. CartPole has two actions in
                      its action space, so in this case n_inputs would be two.

        Returns: the initialised population that is ready for use.
        """
        creature = Creature(n_inputs, n_outputs)
        population = [creature.copy() for _ in range(self.n_pops)]

        return population

    @property
    def champ(self):
        """The best performing creature, who is an all-round champ."""
        self.population = sorted(self.population)

        return self.population[-1]

    @property
    def chump(self):
        """The worst performing creature, who is an all-round chump."""
        self.population = sorted(self.population)

        return self.population[0]

    def train(self, n_episodes=100, n_steps=200, debug_mode=False):
        """Train species of individuals.

        Arguments:
            n_episodes: The number of episodes to trian for.
            n_steps: The maximum number of steps per individual per episode.
            debug_mode: If set to True, some features that aren't intended for
                        testing environments and such are disabled.
        """
        sim_start = time()

        episode_complete_msg_format = "\r{:03d}/{:03d} - " \
                                      "mean fitness: {:.2f} - " \
                                      "median fitness: {:.2f} - " \
                                      "mean time per creature: {:02.4f}s - " \
                                      "total time: {:.4f}s"

        fitness_history = []
        species_history = []
        creature_history = []

        for episode in range(n_episodes):
            episode_start = time()
            fitness_history.append([])
            species_history.append([(species.name, len(species))
                                    for species in self.species])

            self.population = sorted(self.population, key=lambda c: c.composite_fitness)
            worst = self.population[0].copy()
            median = self.population[len(self.population) // 2].copy()
            best = self.population[-1].copy()
            creature_history.append((worst, median, best))

            print('Episode {:02d}/{:02d}'.format(episode + 1, n_episodes))

            for pop_i, creature in enumerate(self.population):
                observation = self.env.reset()

                for step in range(n_steps):
                    action = creature.get_action(observation)
                    observation, reward, done, _ = self.env.step(action)

                    if done:
                        creature.fitness = step + 1

                        break
                else:
                    creature.fitness = n_steps

                fitness_history[episode].append(creature.fitness)
                mean_fitness = np.mean(fitness_history[episode])
                median_fitness = np.median(fitness_history[episode])
                episode_time = time() - episode_start
                mean_time_per_creature = episode_time / (pop_i + 1)

                print(episode_complete_msg_format.format(pop_i + 1, self.n_pops,
                                                         mean_fitness, median_fitness,
                                                         mean_time_per_creature,
                                                         episode_time),
                      end='')

            if episode >= 100 and \
                    np.mean(fitness_history[episode - 100:episode]) >= 195.0:
                print('\nSolved in %d episodes :)' % (episode + 1))
                break

            print()
            self.do_the_thing()
            print('Total episode time: %.4fs.\n' % (time() - episode_start))
        else:
            print('Could not solve in %d episodes :(' % n_episodes)

        print('Total run time: {:.2f}s'.format(time() - sim_start))
        print()

        self.post_training_stuff(debug_mode)

    def post_training_stuff(self, debug_mode=False):
        """Do post training stuff."""
        print('Here are the species that made it to the end and the number of '
              'creatures in each of them:')

        for species in sorted(self.species, key=lambda s: s.name):
            print('%s (%s) - %d creatures - %d generations old.' %
                  (species, species.representative.scientific_name,
                   len(species), species.age))

        best_species = max(self.species, key=lambda s: s.mean_fitness)

        print()
        oldest_creature = max(self.population, key=lambda c: c.age)
        print('The oldest creature was %s, who lived for %d generations.' %
              (oldest_creature, oldest_creature.age))

        print('Out of these species, the best species was %s.' % best_species)
        print('The overall champion was %s who had %d nodes and %d '
              'connections in its neural network.' %
              (best_species.champion,
               len(best_species.champion.phenotype.nodes),
               len(best_species.champion.phenotype.connections)))

        print()

        print('Checking if %s makes the grade...' % best_species.champion,
              end='')
        makes_the_grade = self.makes_the_grade(best_species.champion)
        print(('\r%s makes the grade :)' if makes_the_grade else
               '\r%s doesn\'t make the grade :(') % best_species.champion)
        print()

        if not debug_mode:
            print("Recording %s" % best_species.champion)
            self.record_video(best_species.champion)

        self.env.close()

    def makes_the_grade(self, creature, n_trials=100, passing_grade=195.0):
        """Check if the creature 'passes' the environment.

        Returns: True if the creature passes, False otherwise.
        """
        avg_reward = 0
        env = self.env

        for episode in range(n_trials):
            observation = env.reset()

            episode_reward = 0

            for step in range(200):
                action = creature.get_action(observation)
                observation, reward, done, _ = env.step(action)

                episode_reward += reward

                if done:
                    break

            avg_reward += episode_reward

        return (avg_reward / n_trials) >= passing_grade

    def record_video(self, creature):
        """Record a video of the creature trying to solve the problem.

        Arguments:
            creature: the creature to record.
        """
        env = wrappers.Monitor(self.env, './data/videos/%s' % time())

        for i_episode in range(20):
            observation = env.reset()

            for step in range(200):
                env.render()

                action = creature.get_action(observation)
                observation, _, done, _ = env.step(action)

                if done:
                    print("Episode finished after {} timesteps".format(step + 1))
                    break

    def do_the_thing(self):
        """Do the post-episode stuff such as speciating, adjusting creature
        fitness, crossover etc.
        """
        self.speciate()
        self.adjust_fitness()
        self.blame()
        self.praise()
        self.allot_offspring_quota()
        self.not_so_natural_selection()
        self.mating_season()
        self.spring_cleaning()

    def speciate(self):
        """Place creatures in the population into a species, or create a new
        species if no suitable species exists.
        """
        print('Segregating Communities...', end='')

        for creature in self.population:
            for species_i in self.species:
                if creature.distance(species_i.representative) < \
                        Species.compatibility_threshold:
                    species_i.add(creature)

                    break
            else:
                new_species = Species()
                new_species.add(creature)
                new_species.representative = creature

                self.species.add(new_species)

    def adjust_fitness(self):
        """Adjust the fitness of the population."""
        print('\r' + ' ' * 80, end='')
        print('\rAdjusting good boy points...')

        for creature in self.population:
            creature.adjust_fitness()

    def blame(self):
        """Blame the worst performing individual for preventing the algorithm
        from converging, what a chump.
        """
        print('Blame: %s - fitness: %d (adjusted: %.2f)' %
              (self.chump, self.chump.raw_fitness, self.chump.fitness))

    def praise(self):
        """Praise the best individual for being the best, what a champ."""
        print('Praise: %s - fitness: %d (adjusted: %.2f)' %
              (self.champ, self.champ.raw_fitness, self.champ.fitness))

    def allot_offspring_quota(self):
        """Allot the number of offspring each species is allowed for the
        current generation.

        Sort of like the One-Child policy but we force each species to have
        exactly the amount of babies we tell them to. No more, no less.
        """
        print('\r' + ' ' * 80, end='')
        print('\rAllotting offspring Quota...', end='')
        species_mean_fitness = \
            [species.mean_fitness for species in self.species]
        sum_mean_species_fitness = sum(species_mean_fitness)

        for species, mean_fitness in zip(self.species, species_mean_fitness):
            species.allotted_offspring_quota = \
                int(mean_fitness / sum_mean_species_fitness * self.n_pops)

    def not_so_natural_selection(self):
        """Perform selection on the population.

        Species champions (the fittest creature in a species) are carried over
        to the next generation (elitism). The worst performing portion of
        each species is culled. R.I.P.
        """
        print('\r' + ' ' * 80, end='')
        print('\rEnforcing Survival of the Fittest...', end='')
        new_population = []

        for species in self.species:
            survivors = species.cull_the_weak(NeatAlgorithm.survival_threshold)
            new_population += survivors

        for creature in new_population:
            creature.age += 1

        self.population = new_population

    def mating_season(self):
        """It is now time for mating season, time to make some babies.

        Replace the population with the next generation. Rip last generation.
        """
        print('\r' + ' ' * 80, end='')
        print('\rRearing the next generation...', end='')
        ranked_species = sorted(self.species,
                                key=lambda s: s.champion.composite_fitness)
        best_species = ranked_species[-1]
        the_champ = best_species.champion
        new_population = []

        total_expected_offspring = sum([species.allotted_offspring_quota
                                        for species in self.species])

        if total_expected_offspring < self.n_pops:
            pop_deficit = self.n_pops - total_expected_offspring

            best_species.allotted_offspring_quota += pop_deficit
            total_expected_offspring += pop_deficit
        elif total_expected_offspring > self.n_pops:
            pop_surplus = self.n_pops - total_expected_offspring
            best_species.allotted_offspring_quota -= pop_surplus
            total_expected_offspring -= pop_surplus

        for species in self.species:
            new_population += species.next_generation(the_champ,
                                                      self.population)
            print('.', end='')

        self.population = new_population
        print()

    def spring_cleaning(self):
        """Clean out all the cobwebs and extinct species."""
        self.species = set(filter(lambda s: not s.is_extinct, self.species))

    def to_json(self):
        """Encode the current state of the algorithm as JSON.

        This saves pretty much everything from parameters to individual
        creatures.

        Returns: the generated JSON.
        """
        return dict(
            species=[species.to_json() for species in self.species],
            env=self.env.unwrapped.spec.id,
            n_pops=self.n_pops,
            settings=dict(
                survival_threshold=NeatAlgorithm.survival_threshold,
                compatibility_threshold=Species.compatibility_threshold,
                p_interspecies_mating=Species.p_interspecies_mating,
                disjointedness_importance=Creature.disjointedness_importance,
                excessivity_importance=Creature.excessivity_importance,
                weight_unsameness_importance=Creature.weight_unsameness_importance,
                p_mate_only=Creature.p_mate_only,
                p_mutate_only=Creature.p_mutate_only,
                p_mate_average=Genome.p_mate_average,
                p_mate_choose=Genome.p_mate_choose,
                p_add_node=Genome.p_add_node,
                p_add_connection=Genome.p_add_connection,
                p_recurrent_connection=Genome.p_recurrent_connection,
                p_re_enable_connection=Genome.p_re_enable_connection,
                p_perturb=Genome.p_perturb,
                perturb_range=Genome.perturb_range
            )
        )

    @staticmethod
    def from_json(config):
        """Load an instance of the NEAT algorithm from JSON.

        Arguments:
            config: the JSON dictionary loaded from file.

        Returns: an instance of the NEAT algorithm.
        """
        env = gym.make(config['env'])
        algo = NeatAlgorithm(env)
        algo.species = set(Species.from_json(s_config)
                           for s_config in config['species'])

        algo.population = []

        for species in algo.species:
            algo.population += species.members

        algo.n_pops = config['n_pops']
        NeatAlgorithm.set_config(config['settings'])

        return algo

    @staticmethod
    def set_config(config):
        """Set the parameters of the NEAT algorithm.

        Arguments:
            config: A dictionary containing the key-value pairs for the
                    algorithm parameters.
        """
        NeatAlgorithm.survival_threshold = config['survival_threshold']
        Species.compatibility_threshold = config['compatibility_threshold']
        Species.p_interspecies_mating = config['p_interspecies_mating']
        Creature.disjointedness_importance = config['disjointedness_importance']
        Creature.excessivity_importance = config['excessivity_importance']
        Creature.p_mate_only = config['p_mate_only']
        Creature.p_mutate_only = config['p_mutate_only']
        Genome.p_mate_average = config['p_mate_average']
        Genome.p_mate_choose = config['p_mate_choose']
        Genome.p_add_node = config['p_add_node']
        Genome.p_add_connection = config['p_add_connection']
        Genome.p_recurrent_connection = config['p_recurrent_connection']
        Genome.p_re_enable_connection = config['p_re_enable_connection']
        Genome.p_perturb = config['p_perturb']
        Genome.perturb_range = config['perturb_range']
