from dataclasses import dataclass, field
import numpy as np
import random
from common_elements import PropagatedElement

@dataclass
class Individual(PropagatedElement):
    def __post_init__(self):
        """
        Generate a random initial state if not provided and calculate its cost.
        """
        if not self.state:
            self.state = self.generate_random_genes()
        self.score = self.calculate_cost()

    def generate_random_genes(self):
        """
        Generate a random initial state.
        """
        lu_to_km_coeff = 389703
        tu_to_s_coeff = 382981
        pos_limits = 250 / lu_to_km_coeff
        vel_limits = 0.1 / (lu_to_km_coeff / tu_to_s_coeff)

        position = [np.random.uniform(-pos_limits, pos_limits) for _ in range(3)]
        velocity = [np.random.uniform(-vel_limits, vel_limits) for _ in range(3)]
        self.state = position + velocity


    def mutate(self, mutation_rate: float=0.01, option: bool=False):
        """
        Two mutate options are available:
        - option 1: after implementation add describtion
        - option 2: after implementation add describtion
        """

        if option:
            self.state = self._mutate_1(mutation_rate)
        else:
            self.state = self._mutate_2(mutation_rate)
        pass


    def _mutate_1(self, mutation_rate: float=0.01):
        """
        One of the two mutate options.
        """
        pass

    def _mutate_2(self, mutation_rate: float=0.01):
        """
        One of the two mutate options.
        """
        pass

    def crossover(self, other):
        """
        How to crossover two individuals.
        """
        pass


@dataclass
class Population:
    """
    Main idea list of Individuals.
    """
    size: int
    mutation_rate: float = 0.01
    crossover_rate: float = 0.7
    individuals: list[Individual] = field(default=None, init=False)


    def __post_init__(self):
        """
        If no individuals are provided, create a population of random individuals.
        """
        self.individuals = self.initialize_population()

    def initialize_population(self, size: int = 100):
        """
        Create a population of random individuals with specified size.
        """
        for _ in range(size):
            self.individuals.append(Individual())
        return self.individuals

    def evaluate_population(self):
        """
        Evaluate the cost of each individual in the population.
        """
        pass

    def select_parents(self, option: bool=False):
        """
        Two select options are available:
        - option 1: after implementation add describtion
        - option 2: after implementation add describtion
        """
        if option:
            return self._select_parents_1()
        else:
            return self._select_parents_2()

    def _select_parents_1(self):
        """ One of the two select options. """
        return random.choice(self.individuals)

    def _select_parents_2(self):
        """ One of the two select options. """
        pass

    def crossover(self, parent1: Individual, parent2: Individual, option: bool=False):
        """
        Two select options are available:
        - option 1: after implementation add describtion
        - option 2: after implementation add describtion
        """
        if option:
            return self._crossover_1(parent1, parent2)
        else:
            return self._crossover_2(parent1, parent2)

    def _crossover_1(self, parent1: Individual, parent2: Individual):
        """ One of the two select options. """
        pass

    def _crossover_2(self, parent1: Individual, parent2: Individual):
        """ One of the two select options. """
        pass


    def mutate(self, individual: Individual):
        """
        Mutate an individual.
        """
        pass

    def evolve(self):
        """
        Create a new generation.
        """
        self.evaluate_population()
        new_generation = []

        while len(new_generation) < self.size:
            parent1 = self.select_parents()
            parent2 = self.select_parents()
            offspring = self.crossover(parent1, parent2)
            self.mutate(offspring)
            new_generation.append(offspring)

        self.individuals = new_generation


@dataclass
class GeneticAlgorithm:
    population_size: int
    max_generations: int
    mutation_rate: float = 0.01
    crossover_rate: float = 0.7
    population: Population = field(init=False)

    def __post_init__(self):
        self.population = Population(
            size=self.population_size,
            mutation_rate=self.mutation_rate,
            crossover_rate=self.crossover_rate
        )

    def run(self):
        """
        Main function of the algorithm.
        """
        pass