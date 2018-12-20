import argparse
import random

import gym

from neat.main import NeatAlgorithm


def main(debug_mode=False):
    n_episodes = 100
    n_steps = 200
    n_pops = 150

    if not debug_mode:
        parser = argparse.ArgumentParser(
            description='Run the NEAT genetic algorithm on the CartPole problem.')
        parser.add_argument('--n-episodes', type=int, default=100,
                            help='how many episodes to run the simulation '
                                 'for.')
        parser.add_argument('--n-steps', type=int, default=200,
                            help='the maximum number of steps to run each '
                                 'episode for.')
        parser.add_argument('--n-pops', type=int, default=150,
                            help='how many individuals to have in the '
                                 'population.')
        parser.add_argument('--debug', action='store_true',
                            help='Flag to indicate if NEAT should be run in '
                                 'debug mode.')

        args = parser.parse_args()

        if args.debug:
            debug_mode = True

            n_episodes = args.n_episodes
            n_steps = args.n_steps
            n_pops = args.n_pops

    if debug_mode:
        random.seed(42)

    env = gym.make('CartPole-v0')
    neat = NeatAlgorithm(env, n_pops)
    neat.train(n_episodes, n_steps, debug_mode=debug_mode)


if __name__ == '__main__':
    main()
