from neat.species import Species


class Population:
    """Abstracts the creatures of NEAT."""

    # Percentage of each species that is allowed to reproduce.
    survival_threshold = 0.3

    def __init__(self, seed_creature=None, n_pops=150):
        """Create a population for NEAT.

        Arguments:
            seed_creature: The creature that will be used to creature the
                           initial generation.
            n_pops: How many creatures should be in the population.
        """
        self.n_pops = n_pops
        self.species = set()

        if seed_creature:
            self.creatures = [seed_creature.copy() for _ in range(n_pops)]

            genesis_species = Species()
            genesis_species.assign_members(self.creatures)
            self.species.add(genesis_species)
        else:
            self.creatures = []

    @property
    def champ(self):
        """The best performing creature, who is an all-round champ."""
        self.creatures = sorted(self.creatures)

        return self.creatures[-1]

    @property
    def lukewarm(self):
        """Mr. Luke Warm, neither the worst or the best creature, he is just in
        the middle.

        Returns: the creature with the median composite fitness.
        """
        self.creatures = sorted(self.creatures)

        return self.creatures[len(self.creatures) // 2]

    @property
    def chump(self):
        """The worst performing creature, who is an all-round chump."""
        self.creatures = sorted(self.creatures)

        return self.creatures[0]

    @property
    def oldest_creature(self):
        return sorted(self.creatures, key=lambda c: c.age)[-1]

    @property
    def best_species(self):
        return sorted(self.species, key=lambda s: s.mean_fitness)[-1]

    def speciate(self):
        """Place creatures in the creatures into a species, or create a new
        species if no suitable species exists.
        """
        print('Segregating Communities...', end='')
        # Adding these lines slows down convergence a lot.
        for species in self.species:
            species.members.clear()

        for creature in self.creatures:
            for species in self.species:
                if creature.distance(species.representative) < \
                        Species.compatibility_threshold:
                    species.add(creature)

                    break
            else:
                new_species = Species()
                new_species.add(creature)
                new_species.representative = creature

                self.species.add(new_species)

        self.species = set(filter(lambda s: len(s) > 0, self.species))

    def adjust_fitness(self):
        """Adjust the fitness of the creatures."""
        print('\r' + ' ' * 80, end='')
        print('\rAdjusting good boy points...')

        for creature in self.creatures:
            creature.adjust_fitness()

    def make_history(self):
        """Time to make some history."""
        print('Blame: %s - fitness: %d (adjusted: %.2f)' %
              (self.chump, self.chump.raw_fitness, self.chump.fitness))
        print('Meh: %s - fitness: %d (adjusted: %.2f)' %
              (self.lukewarm, self.lukewarm.raw_fitness, self.lukewarm.fitness))
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

        best_species = self.best_species

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

        print()

    def not_so_natural_selection(self):
        """Perform selection on the creatures.

        Species champions (the fittest creature in a species) are carried over
        to the next generation (elitism). The worst performing portion of
        each species is culled. R.I.P.
        """
        print('\r' + ' ' * 80, end='')
        print('\rEnforcing Survival of the Fittest...', end='')
        new_population = []

        for species in self.species:
            survivors = species.cull_the_weak(Population.survival_threshold)
            new_population += survivors

        for creature in new_population:
            creature.age += 1

        self.creatures = new_population

    def mating_season(self):
        """It is now time for mating season, time to make some babies.

        Replace the creatures with the next generation. Rip last generation.
        """
        print('\r' + ' ' * 80, end='')
        print('\rRearing the next generation...', end='')
        ranked_species = sorted(self.species,
                                key=lambda s: s.champion.composite_fitness)
        best_species = ranked_species[-1]
        the_champ = best_species.champion
        new_population = []

        for species in self.species:
            new_population += species.next_generation(the_champ,
                                                      self.creatures)
            print('.', end='')

        self.creatures = new_population
        self.species = set(filter(lambda s: not s.is_extinct, self.species))
        print()

    def next_generation(self):
        self.speciate()
        self.adjust_fitness()
        self.allot_offspring_quota()
        self.make_history()
        self.not_so_natural_selection()
        self.mating_season()

    def list_species(self):
        for species in sorted(self.species, key=lambda s: s.name):
            print('%s (%s) - %d creatures - %d generations old.' %
                  (species, species.representative.scientific_name,
                   len(species), species.age))

    def to_json(self):
        """Encode the current state of the algorithm as JSON.

        This saves pretty much everything from parameters to individual
        creatures.

        Returns: the generated JSON.
        """
        return dict(
            species=[species.to_json() for species in self.species],
            n_pops=self.n_pops
        )

    @staticmethod
    def from_json(config):
        """Load an instance of the NEAT algorithm from JSON.

        Arguments:
            config: the JSON dictionary loaded from file.

        Returns: an instance of the NEAT algorithm.
        """
        population = Population(None, config['n_pops'])
        population.species = set(Species.from_json(s_config)
                                 for s_config in config['species'])

        population.creatures = []

        for species in population.species:
            population.creatures += species.members

        return population

    def __len__(self):
        return len(self.creatures)
