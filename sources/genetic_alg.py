from dataclasses import dataclass, field
import numpy as np
import random
from common_elements import PropagatedElement, ModelProperties
from constant import DATA_PATH_ORBIT_L2_7


class Individual(PropagatedElement):
    def __init__(self, state=None, model=ModelProperties(filepath=DATA_PATH_ORBIT_L2_7, number_of_measurements=35)):
        super().__init__(state, model)
        if self.state is None:
            self.generate_random_genes()
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


    def _mutate_1(self, mutation_rate: float=0.01):
        """
        One of the two mutate options.
        Method that mutates a single gene with constant shift.
        """
        mutate_index = random.randint(0, 5)
        shift = 0.001
        self.state[mutate_index]+=shift
        return self.state


    def _mutate_2(self, mutation_rate: float=0.01):
        """
        One of the two mutate options.
        Method that mutates random numbers genes with random shift.
        """
        indexes = random.sample([0, 1, 2, 3, 4, 5], k=random.randint(2, 6))
        for i in indexes:
            shift = random.uniform(-0.001,0.001)
            self.state[i]+=shift
        return self.state



    def crossover(self, other, crossover_1: bool = True):
        """
        How to crossover two individuals.
        """


        if crossover_1:
            offspring1_poz_idx = random.choices([1, 2], k=6)
            while offspring1_poz_idx == [1] * 6 or offspring1_poz_idx == [2] * 6:
                offspring1_poz_idx = random.choices([1, 2], k=6)
            offspring2_poz_idx = [2 if x == 1 else 1 for x in offspring1_poz_idx]
        else:
            point = random.randint(1, 5)
            offspring1_poz_idx = self.state[:point] + other.state[point:]
            offspring2_poz_idx = self.state[point:] + other.state[:point]

        offspring1_poz = [self.state[i] if par == 1 else other.state[i] for i, par in enumerate(offspring1_poz_idx)]
        offspring2_poz = [self.state[i] if par == 1 else other.state[i] for i, par in enumerate(offspring2_poz_idx)]
        return offspring1_poz, offspring2_poz

    def crossover_copy(self, other):
        offspring1_poz = self.state
        offspring2_poz = other.state
        return offspring1_poz, offspring2_poz

@dataclass
class Population:
    """
    Main idea list of Individuals.
    """
    size: int
    mutation_rate: float = 0.01
    crossover_rate: float = 0.7
    individuals: list[Individual] = field(default=list, init=False)
    parent_list: list[tuple[Individual, Individual]] = field(default=list, init=False)


    def __post_init__(self):
        """
        If no individuals are provided, create a population of random individuals.
        """
        self.individuals = self.initialize_population()

    def initialize_population(self):
        """
        Create a population of random individuals with specified size.
        """
        return [Individual() for _ in range(self.size)]

    def evaluate_population(self):
        """
        Evaluate the cost of each individual in the population.
        """
        for individual in self.individuals:
            individual.fitness = individual.calculate_cost()

    def select_parents(self, option: bool=False):
        """
        Two select options are available:
        - option 1: after implementation add describtion
        - option 2: after implementation add describtion
        """
        if option:
            self.parent_list = self._select_parents_1()
        else:
            self.parent_list = self._select_parents_2()

    def _select_parents_1(self):
        """ One of the two select options. """
        return random.choice(self.individuals)

    def _select_parents_2(self):
        """ One of the two select options. """
        return random.choice(self.individuals)

    def crossover(self, parent1: Individual, parent2: Individual) -> []:
        """
        Two select options are available:
        - probability 0% - 35%: Inheritance of some features from one parent and others from the other parent.
        - probability 35% - 70%: Random features inheritance.
        - probability 70% - 100%: No crossover - parents copy.
        """
        probability = random.random()
        if probability < self.crossover_rate/2:
            return self._crossover_1(parent1, parent2)
        elif self.crossover_rate/2 <= probability < self.crossover_rate:
            return self._crossover_2(parent1, parent2)
        else:
            offspring1_poz, offspring2_poz = parent1.crossover_copy(parent2)
            return Individual(offspring1_poz), Individual(offspring2_poz)

    def _crossover_1(self, parent1: Individual, parent2: Individual):
        """ One of the two select options. """
        offspring1_poz, offspring2_poz = parent1.crossover(parent2)
        return Individual(offspring1_poz), Individual(offspring2_poz)

    def _crossover_2(self, parent1: Individual, parent2: Individual):
        """ One of the two select options. """
        offspring1_poz, offspring2_poz = parent1.crossover(parent2, False)
        return Individual(offspring1_poz), Individual(offspring2_poz)

    def mutate(self, individual: Individual):
        """
        Mutate an individual.
        """
        if random.random() < self.mutation_rate:
            individual.mutate()

    def evolve(self):
        """
        Create a new generation.
        """
        self.evaluate_population()
        new_generation = []

        while len(new_generation) < self.size:
            parent1 = self.select_parents()
            parent2 = self.select_parents()
            offspring1, offspring2 = self.crossover(parent1, parent2)
            self.mutate(offspring1)
            self.mutate(offspring2)
            new_generation.append(offspring1)
            new_generation.append(offspring2)
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
        for generation in range(self.max_generations):
            self.population.evolve()
            best_individual = max(self.population.individuals, key=lambda ind: ind.score)

if __name__ == "__main__":
    ga = GeneticAlgorithm(population_size=10, max_generations=2)
    ga.run()