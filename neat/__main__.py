import argparse

import gym

from neat.main import NeatAlgorithm


def main():
    parser = argparse.ArgumentParser(
        description='Run the NEAT genetic algorithm on the CartPole problem.')
    parser.add_argument('--n-episodes', type=int, default=100,
                        help='how many episodes to run the simulation for.')
    parser.add_argument('--n-steps', type=int, default=200,
                        help='the maximum number of steps to run each episode for.')
    parser.add_argument('--n-pops', type=int, default=150,
                        help='how many individuals to have in the population.')

    args = parser.parse_args()
    env = gym.make('CartPole-v0')
    neat = NeatAlgorithm(env, args.n_pops)
    neat.train(args.n_episodes, args.n_steps)


if __name__ == '__main__':
    main()
