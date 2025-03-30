from dataclasses import dataclass, field
from copy import deepcopy
import numpy as np
import random
from sources.common_elements import PropagatedElement, ModelProperties
from sources.constant import DATA_PATH_ORBIT_L2_7


class Individual(PropagatedElement):
    def __init__(self, state=None, model=None):
        if model is None:
            model = ModelProperties(filepath=DATA_PATH_ORBIT_L2_7, number_of_measurements=35)
        super().__init__(state, model)

        if self.state is None:
            self.generate_random_genes()

        self.model = model
        self.calculate_cost()

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
        initial_random = position + velocity
        self.state = list(self.model.initial_state + initial_random)

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

    def crossover_1(self, other):
        """
        Random crossover: Each gene has a chance to come from either parent.
        Avoids copying all genes from one parent.
        """
        offspring1_poz_idx = random.choices([1, 2], k=6)
        while offspring1_poz_idx == [1] * 6 or offspring1_poz_idx == [2] * 6:
            offspring1_poz_idx = random.choices([1, 2], k=6)
        offspring2_poz_idx = [2 if x == 1 else 1 for x in offspring1_poz_idx]

        offspring1_poz = [self.state[i] if par == 1 else other.state[i] for i, par in enumerate(offspring1_poz_idx)]
        offspring2_poz = [self.state[i] if par == 1 else other.state[i] for i, par in enumerate(offspring2_poz_idx)]
        return offspring1_poz, offspring2_poz

    def crossover_2(self, other):
        """
        Single-point crossover: Genes are exchanged at a random point.
        """
        point = random.randint(1, 5)
        offspring1_poz = self.state[:point] + other.state[point:]
        offspring2_poz = other.state[:point] + self.state[point:]
        return offspring1_poz, offspring2_poz

    def crossover_copy(self, other):
        """
         No crossover: Offspring are direct copies of the parents.
        """
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
    model: ModelProperties = field(default_factory=lambda: ModelProperties(filepath=DATA_PATH_ORBIT_L2_7, number_of_measurements=35))
    individuals: list[Individual] = field(default_factory=list, init=False)


    def __post_init__(self):
        """
        If no individuals are provided, create a population of random individuals.
        """
        self.individuals = self.initialize_population()

    def initialize_population(self):
        """
        Create a population of random individuals with specified size.
        """
        individuals = [Individual(model=self.model) for _ in range(self.size)]
        return individuals

    def evaluate_population(self):
        """
        Evaluate the cost of each individual in the population.
        """
        for individual in self.individuals:
            individual.fitness = individual.calculate_cost()

    def select_parents(self, option: bool=True, tournament_size:int = None):
        """
        Two select options are available:
        - option 1: after implementation add describtion
        - option 2: after implementation add describtion
        """
        if option:
            return self._tournament_selection(tournament_size)
        else:
            return self._roulette_selection()

    def _tournament_selection(self, tournament_size: int):
        """ One of the two select options. """
        copy_list_of_individuals = deepcopy(self.individuals)
        if tournament_size is None:
            tournament_size = self.size // 10
        tournament_1 = random.sample(copy_list_of_individuals, tournament_size)
        parent_1 = min(tournament_1, key=lambda ind: ind.score)
        copy_list_of_individuals.remove(parent_1)
        tournament_2 = random.sample(copy_list_of_individuals, tournament_size)
        parent_2 = min(tournament_2, key=lambda ind: ind.score)
        return parent_1, parent_2

    def _roulette_selection(self):
        """ One of the two select options. """
        copy_list_of_individuals = deepcopy(self.individuals)
        total_fitness = self.sum_of_fittness()
        if total_fitness == 0:
            return tuple(random.sample(copy_list_of_individuals, 2)) 

        selection_probs_1 = [individual.score / total_fitness for individual in copy_list_of_individuals]
        selected_index_1 = random.choices(range(len(copy_list_of_individuals)), weights=selection_probs_1, k=1)[0]
        parent_1 = copy_list_of_individuals.pop(selected_index_1)
        selection_probs_2 = [individual.score / total_fitness for individual in copy_list_of_individuals]
        selected_index_2 = random.choices(range(len(copy_list_of_individuals)), weights=selection_probs_2, k=1)[0]
        parent_2 = copy_list_of_individuals.pop(selected_index_2)
        return parent_1, parent_2

    def crossover(self, parent1: Individual, parent2: Individual) -> []:
        """
        Two select options are available:
        - probability 0% - 35%: Inheritance of some features from one parent and others from the other parent.
         probability 35% - 70%: Single-point crossover.
        - probability 70% - 100%: No crossover - parents copy.
        :return: The function returns a tuple of two offspring generated by crossover. The type of crossover is determined
                 by the probability and crossover rate. Specifically:
            - If the probability is less than half of the crossover rate, the first crossover method (`_crossover_1`) is applied,
              resulting in an offspring that inherits some features from each parent.
            - If the probability is in the middle range (between half and the full crossover rate), the second crossover method (`_crossover_2`) is used,
              which implements single-point crossover. A random point is selected, and the genetic material of both parents is exchanged at that point.
            - If the probability is greater than or equal to the crossover rate, no crossover occurs, and the parents are copied directly.
              The method `crossover_copy` is used to generate two offspring by simply copying the parents' features.
        """
        probability = random.random()
        if probability < self.crossover_rate/2:
            return self._crossover_1(parent1, parent2)
        elif self.crossover_rate/2 <= probability < self.crossover_rate:
            return self._crossover_2(parent1, parent2)
        else:
            offspring1_poz, offspring2_poz = parent1.crossover_copy(parent2)
            return Individual(offspring1_poz), Individual(offspring2_poz)

    def _crossover_1(self, parent1: Individual, parent2: Individual)-> tuple[Individual, Individual]:
        """ One of the two select options. """
        offspring1_poz, offspring2_poz = parent1.crossover_1(parent2)
        return Individual(offspring1_poz), Individual(offspring2_poz)

    def _crossover_2(self, parent1: Individual, parent2: Individual)-> tuple[Individual, Individual]:
        """ One of the two select options. """
        offspring1_poz, offspring2_poz = parent1.crossover_2(parent2)
        return Individual(offspring1_poz), Individual(offspring2_poz)

    def mutate(self, individual: Individual):
        """
        Mutate an individual.
        """
        if random.random() < self.mutation_rate:
            individual.mutate(random.choice([True, False]))

    def evolve(self, option: bool=True, tournament_size:int = None):
        """
        Create a new generation.
        """
        self.evaluate_population()
        new_generation = []

        while len(new_generation) < self.size:
            parent1, parent2 = self.select_parents(option, tournament_size)
            offspring1, offspring2 = self.crossover(parent1, parent2)
            self.mutate(offspring1)
            self.mutate(offspring2)
            offspring1.calculate_cost()
            offspring2.calculate_cost()
            new_generation.append(offspring1)
            new_generation.append(offspring2)
        self.individuals = new_generation

    def sum_of_fittness(self):
        return sum(individual.score for individual in self.individuals)


def genetic_alg(population_size:int, max_generaton:int, mutation_rate: float = 0.01, crossover_rate: float = 0.7,
                option: bool = True, tournament_size: int = None, file_path: str = DATA_PATH_ORBIT_L2_7,
                number_of_measurements: int = 35):
    best_scores_vector = []
    model = ModelProperties(filepath=file_path, number_of_measurements=number_of_measurements)
    population = Population(size=population_size, mutation_rate=mutation_rate, crossover_rate=crossover_rate, model=model)
    best_individual = None
    best_individual_init = min(population.individuals, key=lambda ind: ind.score)
    best_scores_vector.append(best_individual_init.score)
    for generation in range(max_generaton):
        population.evolve(option, tournament_size)
        print(f"Generation {generation}:")
        print(f"Population: {population}")
        best_individual = min(population.individuals, key=lambda ind: ind.score)
        best_scores_vector.append(best_individual.score)
        print("Best score: ", best_individual.score)
        # potrzebne jest zwracanie takie jak w pso do plotowania
    return [
        best_individual_init,
        best_individual,
        model.chosen_states,
        model.initial_state,
        max_generaton,
        best_scores_vector
    ]

if __name__ == "__main__":
    genetic_alg(80, 30, tournament_size=5, mutation_rate=0.1)