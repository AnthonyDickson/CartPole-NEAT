"""Implements the NEAT algorithm."""
import hashlib
import json
import os
from time import time

import gym
import numpy as np
import requests
from gym import wrappers

from neat.creature import Creature
from neat.genome import Genome
from neat.population import Population
from neat.pso import PSO
from neat.species import Species


class NeatAlgorithm:
    """An implementation of the NEAT algorithm based off the original paper."""

    api_url = 'http://localhost:5000/api'

    def __init__(self, env, n_pops=150):
        self.env = env
        self.n_trials = env.spec.trials
        self.reward_threshold = env.spec.reward_threshold
        genesis = Creature(env.observation_space.shape[0], env.action_space.n)
        self.population = Population(genesis, n_pops)

        self.fitness_history = []
        self.snapshots = []

        run_id_hash = hashlib.sha1()
        run_id_hash.update(str(time()).encode('utf-8'))
        self.run_id = run_id_hash.hexdigest()[:16]

        r = requests.request('POST', NeatAlgorithm.api_url + '/runs',
                             json=dict(id=self.run_id))
        if r.status_code != 201:
            print('WARNING: Could not add run data to database.')
            print(r.json())

    def train(self, n_episodes=100, n_steps=200, n_pso_episodes=5,
              debug_mode=False):
        """Train species of individuals.

        Arguments:
            n_episodes: The number of episodes to trian for.
            n_steps: The maximum number of steps per individual per episode.
            n_pso_episodes: The number of episodes
            debug_mode: If set to True, some features that aren't intended for
                        testing environments and such are disabled.
        """
        sim_start = time()

        episode_complete_msg_format = "\r{:03d}/{:03d} - " \
                                      "mean fitness: {:.2f} - " \
                                      "median fitness: {:.2f} - " \
                                      "mean time per creature: {:02.4f}s - " \
                                      "total time: {:.4f}s"

        for episode in range(n_episodes):
            print('Episode {:02d}/{:02d}'.format(episode + 1, n_episodes))

            episode_start = time()
            self.fitness_history.append([])

            if n_pso_episodes > 0:
                print('Acquiring Collective Intelligence...')
                for species in self.population.species:
                    pso = PSO(self.env, species.members)
                    pso.train(n_episodes=n_pso_episodes, n_steps=n_steps)
                    pso.apply()
                print('\nCollective Intelligence acquired in {:.4f}s.'
                      .format(time() - episode_start))

            print('Evaluating Population Goodness...')
            for pop_i, creature in enumerate(self.population.creatures):
                observation = self.env.reset()

                for step in range(n_steps):
                    action = creature.get_action(observation)
                    observation, reward, done, _ = self.env.step(action)

                    if done:
                        creature.fitness = step + 1

                        break
                else:
                    creature.fitness = n_steps

                self.fitness_history[episode].append(creature.fitness)
                mean_fitness = np.mean(self.fitness_history[episode])
                median_fitness = np.median(self.fitness_history[episode])
                episode_time = time() - episode_start
                mean_time_per_creature = episode_time / (pop_i + 1)

                print(episode_complete_msg_format
                      .format(pop_i + 1, self.population.n_pops, mean_fitness,
                              median_fitness, mean_time_per_creature,
                              episode_time),
                      end='')

            if episode >= self.n_trials and \
                    np.mean(self.fitness_history[episode - self.n_trials:episode]) \
                    >= self.reward_threshold:
                print('\nSolved in %d episodes :)' % (episode + 1))
                break

            print()
            self.do_the_thing()
            print('Total episode time: %.4fs.\n' % (time() - episode_start))
        else:
            print('Could not solve in %d episodes :(' % n_episodes)

        print('Total run time: {:.2f}s'.format(time() - sim_start))
        print()

        r = requests.request('PATCH', '%s/runs/%s/finished' %
                             (NeatAlgorithm.api_url, self.run_id))

        if r.status_code != 204:
            print("WARNING: Was not able to update run finished status.")
            print(r.json())

        self.post_training_stuff(n_steps, debug_mode)

    def post_training_stuff(self, n_steps, debug_mode=False):
        """Do post training stuff."""
        print('Here are the species that made it to the end and the number of '
              'creatures in each of them:')

        self.population.list_species()

        best_species = self.population.best_species

        print()
        oldest_creature = self.population.oldest_creature
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
        makes_the_grade = self.makes_the_grade(best_species.champion, n_steps)
        print(('\r%s makes the grade :)' if makes_the_grade else
               '\r%s doesn\'t make the grade :(') % best_species.champion)
        print()

        if not debug_mode:
            print("Recording %s" % best_species.champion)
            self.record_video(best_species.champion, n_steps=n_steps)

        self.dump()

        self.env.close()

    def makes_the_grade(self, creature, n_steps):
        """Check if the creature 'passes' the environment.

        Returns: True if the creature passes, False otherwise.
        """
        avg_reward = 0
        env = self.env

        for episode in range(self.n_trials):
            observation = env.reset()

            episode_reward = 0

            for step in range(n_steps):
                action = creature.get_action(observation)
                observation, reward, done, _ = env.step(action)

                episode_reward += reward

                if done:
                    break

            avg_reward += episode_reward

        return (avg_reward / self.n_trials) >= self.reward_threshold

    def record_video(self, creature, n_episodes=20, n_steps=200):
        """Record a video of the creature trying to solve the problem.

        Arguments:
            creature: the creature to record.
            n_episodes: how many episodes to record.
            n_steps: how many steps to run each episode for.
        """
        env = wrappers.Monitor(self.env, './data/videos/%s' % time())

        for i_episode in range(n_episodes):
            observation = env.reset()

            for step in range(n_steps):
                env.render()

                action = creature.get_action(observation)
                observation, _, done, _ = env.step(action)

                if done:
                    print("Episode finished after {} timesteps"
                          .format(step + 1))
                    break

    def dump(self, path='data/training/', filename=None):
        """Save training data to file."""
        if path[-1] != '/':
            path += '/'

        fullpath = path + (filename if filename else '%s.json' % time())

        os.makedirs(path, exist_ok=True)

        with open(fullpath, 'w') as f:
            json.dump(self.to_json(), f)

        print('Saved training data to: %s.' % fullpath)

    def do_the_thing(self):
        """Do the post-episode stuff such as speciating, adjusting creature
        fitness, crossover etc.
        """
        self.population.speciate()
        self.population.adjust_fitness()
        self.population.allot_offspring_quota()
        self.population.make_history()
        self.population.not_so_natural_selection()
        self.population.mating_season()
        self.population.spring_cleaning()

    def to_json(self):
        """Encode the current state of the algorithm as JSON.

        This saves pretty much everything from parameters to individual
        creatures.

        Returns: the generated JSON.
        """
        return dict(
            run_id=self.run_id,
            env=self.env.unwrapped.spec.id,
            population=self.population.to_json(),
            settings=dict(
                survival_threshold=Population.survival_threshold,
                compatibility_threshold=Species.compatibility_threshold,
                p_interspecies_mating=Species.p_interspecies_mating,
                disjointedness_importance=Creature.disjointedness_importance,
                excessivity_importance=Creature.excessivity_importance,
                weight_unsameness_importance=
                Creature.weight_unsameness_importance,
                p_mate_only=Creature.p_mate_only,
                p_mutate_only=Creature.p_mutate_only,
                p_mate_average=Genome.p_mate_average,
                p_mate_choose=Genome.p_mate_choose,
                p_add_node=Genome.p_add_node,
                p_add_connection=Genome.p_add_connection,
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
        algo.run_id = config['run_id']
        algo.n_trials = env.spec.trials
        algo.reward_threshold = env.spec.reward_threshold
        algo.population = Population.from_json(config['population'])

        NeatAlgorithm.set_config(config['settings'])

        return algo

    @staticmethod
    def set_config(config):
        """Set the parameters of the NEAT algorithm.

        Arguments:
            config: A dictionary containing the key-value pairs for the
                    algorithm parameters.
        """
        Population.survival_threshold = config['survival_threshold']
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
        Genome.p_re_enable_connection = config['p_re_enable_connection']
        Genome.p_perturb = config['p_perturb']
        Genome.perturb_range = config['perturb_range']
