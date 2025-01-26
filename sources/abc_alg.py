"""
Artificial Bee Colony Algorithm implementation
"""
from copy import deepcopy
import random
import numpy as np
from sources.common_elements import PropagatedElement, ModelProperties, Swarm
from sources.data_structures import MandatorySettingsABC, OptionalSettingsABC


class SwarmABC(Swarm):
    """
    Creates a swarm of food sources (elements).
    Defines its basic characteristics.
    """

    def generate_initial_population(self,
                                    opt_if_two_stage_pso=0,
                                    opt_best_velocity=None):
        """
        Generates a new population of Food class objects, based on the initial points
        that are passed as an argument.
        """
        initial_random = super().generate_initial_population(opt_if_two_stage_pso, opt_best_velocity)
        for idx in range(self.population_size):
            initial_conditions = initial_random[idx]
            food_object = Food(initial_conditions, self, self.model)
            self.elements.append(food_object)


class Food(PropagatedElement):
    """
    Defines a food source (the fundamental concept of ABC algorithm) and all its properties.
    """

    def __init__(self, state, swarm, model):
        super().__init__(state, model)
        self.swarm = swarm
        self.probability = 0
        self.neighbours = []
        self.active_flag = False
        self.inactive_cycles = 0

    def generate_neighbours(
            self,
            number_of_neighbours,
            limits,
            neighbourhood_type=0,
            dim_probability=0.5):
        """
        Clear the current list of neighbours and generate <number_of_neighbours> new ones.
        :param number_of_neighbours: int value
        :param limits tolerance value with respect to position [km] and velocity [km/s]
        :param neighbourhood_type: parameter setting the way neighbourhood is generated
        :param dim_probability: defines the probability of a dimention's value change
        :return: self.neighbours: refreshed neighbours
        """
        neighbours_pos_limits, neighbours_vel_limits = limits
        lu_to_km_coeff = 389703
        tu_to_s_coeff = 382981
        pos_limits = neighbours_pos_limits / lu_to_km_coeff
        vel_limits = neighbours_vel_limits / (lu_to_km_coeff / tu_to_s_coeff)

        self.neighbours = []

        if neighbourhood_type == 0:
            for _ in range(number_of_neighbours):
                new_state = np.zeros(6)
                for dim in range(3):
                    new_state[dim] = self.state[dim] + \
                                     random.uniform(-1 * pos_limits, pos_limits)
                for dim in range(3, 6):
                    new_state[dim] = self.state[dim] + \
                                     random.uniform(-1 * vel_limits, vel_limits)
                new_food_source = Food(new_state, self.swarm, self.model)
                new_food_source.calculate_cost()
                self.neighbours.append(new_food_source)

        if neighbourhood_type == 1:
            accepted_dims = [dim for dim in range(
                6) if random.random() < dim_probability]
            for _ in range(number_of_neighbours):
                new_state = self.state
                for dim in accepted_dims:
                    if dim < 3:
                        new_state[dim] += random.uniform(-1 *
                                                         pos_limits, pos_limits)
                    elif dim >= 3:
                        new_state[dim] += random.uniform(-1 *
                                                         vel_limits, vel_limits)

                new_food_source = Food(new_state, self.swarm, self.model)
                new_food_source.calculate_cost()
                self.neighbours.append(new_food_source)

    def choose_best_neighbour(self):
        """
        Choosing the best neighbour source (or itself,
        if any neighbouring source decrease the score)
        """
        source_all_options = [self] + self.neighbours
        source_all_scores = [
            new_source.score for new_source in source_all_options]
        best_source_idx = np.argmin(source_all_scores)
        if best_source_idx != 0:
            self.update(source_all_options[best_source_idx])

    def update(self, new_source):
        """
        Overrides a source properties with the new ones
        :param new_source: updated source (state and score)
        """
        self.state = new_source.state
        self.score = new_source.score
        self.probability = 0
        self.neighbours = []
        self.active_flag = True
        self.inactive_cycles = 0

    def scout_new_source(
            self,
            initial_state,
            generating_method=0,
            vel_limits=0.05):
        """
        Generates new source if a given one no longer offers any improvement.
        :param initial_state: the original trajectory's first point
        :param generating_method: defines the way new source will be generated
        with respect to the initial state
        :param vel_limits: defines the maximum tolerance of new velocity values
        with respect to the initial state
        """
        lu_to_km_coeff = 389703
        tu_to_s_coeff = 382981
        pos_limits = 250 / lu_to_km_coeff
        new_state = []
        if generating_method == 0:
            vel_limits = 0.05 / (lu_to_km_coeff / tu_to_s_coeff)
            new_state = ([np.random.uniform(-1 * pos_limits, pos_limits) for _ in range(3)]
                         + [np.random.uniform(-1 * vel_limits, 1 * vel_limits) for _ in range(3)]
                         + initial_state)

        elif generating_method == 1:
            # Alternative 1: new_state depending on current best global state - the best
            # up-to-date result
            pos_limits = 125 / lu_to_km_coeff
            random_vel_factor = [
                1 + np.random.uniform(-1 * vel_limits, 1 * vel_limits) for _ in range(3)]
            new_state = np.array([np.random.uniform(-1 * pos_limits, pos_limits)
                                + self.swarm.global_best_state[i] for i in range(3)]
                                +[random_vel_factor[i] * self.swarm.global_best_state[i + 3]
                                  for i in range(3)])

        elif generating_method == 2:
            # Alternative 2: new_state depending on initial state (position)
            # and global best state (velocity)
            random_vel_factor = [
                1 + np.random.uniform(-1 * vel_limits, 1 * vel_limits) for _ in range(3)]
            new_state = np.array([np.random.uniform(-1 * pos_limits, pos_limits)
                                + initial_state[i] for i in range(3)]
                                + [random_vel_factor[i] * self.swarm.global_best_state[i + 3]
                                   for i in range(3)])

        new_source = Food(new_state, self.swarm, self.model)
        new_source.calculate_cost()
        self.update(new_source)


def abc_alg(
        mandatory,
        optional=None):
    """
    Heart of ABC algorithm.
    :param mandatory: structure of mandatory parameters, including:
    :> max_iterations: number of iterations
    :> population_size: number of particles
    :> employee_phase_neighbours: number of neighbours visited during employee phase
    :> onlooker_phase_neighbours: number of neighbours visited during onlooker phase
    :> number_of_measurements: number of points where the orbits are compared
    :> neighbours_pos_limits: defines the limits of neighbourhood for sources' positions
    :> neighbours_vel_limits: defines the limits of neighbourhood for sources' velocities
    :> inactive_cycles_limit: number of iterations where no improvement of a source is accepted
    :param optional: structure of optional parameters, including:
    :> inactive_cycles_setter: optional - modificates the number of inactive cycles
    depending on the current iteration
    :> probability_distribution_setter: optional - changes the probability
    the onlooker bees choose the source with
    :> generating_method: optional - defines the way new source is generated
    :> neighbourhood_type: optional - defines the way neighbourhood is handled
    :> neigh_percent: optional - defines the limits of velocity tolerance
    while generating new sources
    :> dim_probability: optional - defines the probability of modyfing a given dimention value
    :>orbit_filepath: path to raw data from NASA database
    :return: set of information about the initial and final solutions
    """

    if optional is None:
        optional = OptionalSettingsABC()

    model = ModelProperties(optional.orbit_filepath, mandatory.number_of_measurements)

    swarm = SwarmABC(
        mandatory.population_size,
        mandatory.max_iterations,
        model)

    swarm.generate_initial_population()

    for source in swarm.elements:
        source.calculate_cost()
    initial_swarm = deepcopy(swarm)
    best_scores_vector = []

    for it in range(mandatory.max_iterations):
        print("iter. no. ", it)
        employee_phase(mandatory, optional, swarm)
        onlooker_phase(mandatory, optional, swarm)
        scout_phase(mandatory, optional, swarm, model.initial_state, it)
        swarm.update_global_best()

        print('global best score: ', swarm.global_best_score)
        best_scores_vector.append(swarm.global_best_score)

    final_swarm = deepcopy(swarm)
    return [
        initial_swarm,
        final_swarm,
        model.chosen_states,
        model.initial_state,
        mandatory.max_iterations,
        best_scores_vector]

def employee_phase(
        mandatory,
        optional,
        swarm):
    """
    # Employee phase
    # Actions:
    # 1. Set all active_flag values to False (new cycle reset).
    # 2. Create new solutions as neighbours of each current food source.
    # 3. Choose the best one, based on costs (the lower, the better). Forget the others.
    :param mandatory: structure of mandatory parameters
    :param optional: structure of optional parameters
    :param swarm: Swarm object with all its elements
    """

    for source in swarm.elements:
        source.active_flag = False
        source.generate_neighbours(
            mandatory.employee_phase_neighbours,
            [mandatory.neighbours_pos_limits,
            mandatory.neighbours_vel_limits],
            optional.neighbourhood_type,
            optional.dim_probability)
        source.choose_best_neighbour()

def onlooker_phase(
        mandatory,
        optional,
        swarm):
    """
    # Onlooker phase
    # Actions:
    # 1. Calculate the probability of choosing the source: the better fit,
    # the higher probability.
    # 2. Based on the mentioned probability, visit a source and try to find a better neighbour.
    :param mandatory: structure of mandatory parameters
    :param optional: structure of optional parameters
    :param swarm: Swarm object with all its elements
    """
    probabilities_vector = [0]

    if optional.probability_distribution_setter == 0:
        score_list = [source.score for source in swarm.elements]
        min_cost = np.min(score_list)
        max_cost = np.max(score_list)
        range_score = min_cost + max_cost
        total_cost = np.sum(
            [range_score - source.score for source in swarm.elements])
        for source in swarm.elements:
            source.probability = (range_score - source.score) / total_cost
            probabilities_vector.append(
                probabilities_vector[-1] + source.probability)

    elif optional.probability_distribution_setter == 1:
        total_cost = np.sum([1 / source.score for source in swarm.elements])
        for source in swarm.elements:
            source.probability = (1 / source.score) / total_cost
            probabilities_vector.append(
                probabilities_vector[-1] + source.probability)

    chosen_state_idx = np.inf  # Temporary assignment
    for _ in range(swarm.population_size):
        probability = random.uniform(0, 1)
        for idx, border_prob in enumerate(probabilities_vector, -1):
            if border_prob > probability:
                chosen_state_idx = idx
                break
        swarm.elements[chosen_state_idx].generate_neighbours(
            mandatory.onlooker_phase_neighbours,
            [mandatory.neighbours_pos_limits,
            mandatory.neighbours_vel_limits],
            optional.neighbourhood_type,
            optional.dim_probability)
        swarm.elements[chosen_state_idx].choose_best_neighbour()

def scout_phase(
        mandatory,
        optional,
        swarm,
        initial_state,
        it):
    """
    # Scout phase
    # Actions:
    # 1. Establish the sources that weren't updated for a given number of iterations.
    # 2. Generate new sources in place of the found inactive ones.
    :param mandatory: structure of mandatory parameters
    :param optional: structure of optional parameters
    :param swarm: Swarm object with all its elements
    :param initial_state: vector of 6 elements representing the initial trajectory's first point
    :param it: iteration number
    """
    if optional.inactive_cycles_setter:
        if it >= 0:
            mandatory.inactive_cycles_limit = 1
        if it >= 50:
            mandatory.inactive_cycles_limit = 3
        if it >= 100:
            mandatory.inactive_cycles_limit = 5
        if it >= 200:
            mandatory.inactive_cycles_limit = 7

    for source in swarm.elements:
        if source.active_flag is False:
            source.inactive_cycles += 1  # Since all updates cause inactive_cycle drop by -1,
            # it will only increase when no update was done
            if source.inactive_cycles >= mandatory.inactive_cycles_limit and it >= 1:
                source.scout_new_source(
                    initial_state, optional.generating_method, optional.neigh_percent)

def main():
    """
    A test instance - will not be run when algorithm used by GUI
    """
    abc_alg(MandatorySettingsABC())


if __name__ == '__main__':
    main()
