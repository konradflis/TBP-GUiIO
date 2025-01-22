from scipy import integrate
import numpy as np
from numpy import linalg
import random
from Sources.data_load import transfer_raw_data_to_trajectory, convert_to_metric_units
from copy import deepcopy


class Swarm:
    """
    Creates a swarm of food sources.
    Defines its basic characteristics.
    """

    def __init__(
            self,
            population_size,
            number_of_measurements,
            max_iterations,
            chosen_states,
            initial_state):
        self.population_size = population_size
        self.number_of_measurements = number_of_measurements
        self.max_iterations = max_iterations
        self.chosen_states = chosen_states
        self.sources = []
        self.global_best_score = np.inf
        self.global_best_state = []
        self.original_trajectory = None
        self.initial_state = initial_state

    def generate_initial_population(self, initial_random):
        """
        Generates a new population of Bee class objects, based on the initial points that are passed as an argument.
        :param initial_random: list of 6-dimensional points representing each particle's initial position and velocity
        :return: None
        """
        for idx in range(self.population_size):
            initial_conditions = initial_random[idx]
            food_object = Food(initial_conditions, self)
            self.sources.append(food_object)

    def convert_to_metric_units(self):
        """
        Converts the LU and LU/TU units to km and km/s respectively
        :return: None
        """
        LU_to_km_coeff = 389703
        TU_to_s_coeff = 382981
        self.global_best_state[:3] = self.global_best_state[:3] * LU_to_km_coeff
        self.global_best_state[3:] = self.global_best_state[3:] * (LU_to_km_coeff / TU_to_s_coeff)


class Food:
    """
    Defines a food source and all its properties.
    """

    def __init__(self, state, swarm):
        self.state = state
        self.swarm = swarm
        self.mu = 0.01215058560962404
        self.score = np.inf
        self.probability = 0
        self.neighbours = []
        self.active_flag = False
        self.inactive_cycles = 0

    def propagate(self, time_vect, time_span):
        """
        Method propagating the initial conditions using the defined differential equations.
        :param time_vect: points in time that are used to compare the original and propagated trajectory
        :param time_span: [0, time_span] is the time limit for the solver
        :return: propagated trajectory, which is a result of solve_ivp with its all atributes
        """
        x, y, z, vx, vy, vz = self.state
        propagated_trajectory = integrate.solve_ivp(
            self.__diff_equation, [
                0, time_span], [
                x, y, z, vx, vy, vz], t_eval=time_vect, method='DOP853')
        return propagated_trajectory

    def __diff_equation(self, t, parameters):
        """
        :param t: function handler parameter
        :param parameters: positions and velocities (6 parameters altogether)
        :return: defined outputs of differential equations (positions and velocities)
        """
        x1, y1, z1, x2, y2, z2 = parameters
        r_se = np.sqrt(((x1 + self.mu) ** 2 + y1 ** 2 + z1 ** 2) ** 3)
        r_sm = np.sqrt(((x1 + self.mu - 1) ** 2 + y1 ** 2 + z1 ** 2) ** 3)

        out_x1, out_x2 = np.array([x2, 2 * y2 + x1 - (1 - self.mu) *
                                   (x1 + self.mu) / r_se - self.mu * (x1 + self.mu - 1) / r_sm])
        out_y1, out_y2 = np.array(
            [y2, -2 * x2 + y1 - (1 - self.mu) * y1 / r_se - self.mu * y1 / r_sm])
        out_z1, out_z2 = np.array(
            [z2, -1 * (1 - self.mu) * z1 / r_se - self.mu * z1 / r_sm])

        return [out_x1, out_y1, out_z1, out_x2, out_y2, out_z2]

    def generate_neighbours(
            self,
            number_of_neighbours,
            neighbours_pos_limits,
            neighbours_vel_limits,
            neighbourhood_type=0,
            dim_probability=0.5):
        """
        Clear the current list of neighbours and generate <number_of_neighbours> new ones.
        :param number_of_neighbours: int value
        :param neighbours_pos_limits: tolerance value [km]
        :param neighbours_vel_limits: tolerance value [km/s]
        :param neighbourhood_type: parameter setting the way neighbourhood is generated
        :param dim_probability: defines the probability of a dimention's value change
        :return: self.neighbours: refreshed neighbours
        """
        LU_to_km_coeff = 389703
        TU_to_s_coeff = 382981
        pos_limits = neighbours_pos_limits / LU_to_km_coeff
        vel_limits = neighbours_vel_limits / (LU_to_km_coeff / TU_to_s_coeff)

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
                new_food_source = Food(new_state, self.swarm)
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

                new_food_source = Food(new_state, self.swarm)
                new_food_source.calculate_cost()
                self.neighbours.append(new_food_source)

    def choose_best_neighbour(self):
        """
        Choosing the best neighbour source (or itself, if any neighbouring source decrease the score)
        :return: None
        """
        source_all_options = [self] + self.neighbours
        source_all_scores = [
            new_source.score for new_source in source_all_options]
        best_source_idx = np.argmin(source_all_scores)
        if best_source_idx != 0:
            self.update(source_all_options[best_source_idx])

    def calculate_cost(self):
        """
        Calculating the cost of each trajectory as a sum of differences between the expected and propagated results in
        :number_of_measurements: points.
        :return: None
        """
        time_vect = np.array(self.swarm.chosen_states['Time (TU)'])
        vect_size = len(time_vect)
        time_span = time_vect[vect_size - 1] + 1
        propagate = self.propagate(time_vect, time_span)
        original_trajectory = self.swarm.original_trajectory.T
        propagated_trajectory = propagate.y

        error = 0
        if propagated_trajectory.shape[1] == self.swarm.number_of_measurements:
            for it in range(self.swarm.number_of_measurements):
                error += linalg.norm(original_trajectory[it,
                                     :3] - propagated_trajectory[:3, it])
            self.score = error
        else:
            self.score = np.inf

    def update(self, new_source):
        """
        Overrides a source properties with the new ones
        :param new_source: updated source (state and score)
        :return: None
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
        :param generating_method: defines the way new source will be generated with respect to the initial state
        :param vel_limits: defines the maximum tolerance of new velocity values with respect to the initial state
        :return: None
        """
        LU_to_km_coeff = 389703
        TU_to_s_coeff = 382981
        pos_limits = 250 / LU_to_km_coeff

        if generating_method == 0:
            vel_limits = 0.05 / (LU_to_km_coeff / TU_to_s_coeff)
            new_state = [np.random.uniform(-1 * pos_limits,
                                           pos_limits),
                         np.random.uniform(-1 * pos_limits,
                                           pos_limits),
                         np.random.uniform(-1 * pos_limits,
                                           pos_limits),
                         np.random.uniform(-1 * vel_limits,
                                           1 * vel_limits),
                         np.random.uniform(-1 * vel_limits,
                                           1 * vel_limits),
                         np.random.uniform(-1 * vel_limits,
                                           1 * vel_limits)] + initial_state

        elif generating_method == 1:
            """Alternative: new_state depending on current best global state - the best up-to-date result"""
            pos_limits = 125 / LU_to_km_coeff
            random_vel_factor = [
                1 + np.random.uniform(-1 * vel_limits, 1 * vel_limits) for _ in range(3)]
            new_state = np.array([np.random.uniform(-1 * pos_limits,
                                                    pos_limits) + self.swarm.global_best_state[0],
                                  np.random.uniform(-1 * pos_limits,
                                                    pos_limits) + self.swarm.global_best_state[1],
                                  np.random.uniform(-1 * pos_limits,
                                                    pos_limits) + self.swarm.global_best_state[2],
                                  random_vel_factor[0] * self.swarm.global_best_state[3],
                                  random_vel_factor[1] * self.swarm.global_best_state[4],
                                  random_vel_factor[2] * self.swarm.global_best_state[5]])

        elif generating_method == 2:
            """Alternative 2: new_state depending on initial state (position) and global best state (velocity)"""
            random_vel_factor = [
                1 + np.random.uniform(-1 * vel_limits, 1 * vel_limits) for _ in range(3)]
            new_state = np.array([np.random.uniform(-1 * pos_limits,
                                                    pos_limits) + initial_state[0],
                                  np.random.uniform(-1 * pos_limits,
                                                    pos_limits) + initial_state[1],
                                  np.random.uniform(-1 * pos_limits,
                                                    pos_limits) + initial_state[2],
                                  random_vel_factor[0] * self.swarm.global_best_state[3],
                                  random_vel_factor[1] * self.swarm.global_best_state[4],
                                  random_vel_factor[2] * self.swarm.global_best_state[5]])

        new_source = Food(new_state, self.swarm)
        new_source.calculate_cost()
        self.update(new_source)


def final_plot(
        initial_swarm,
        final_swarm,
        chosen_states,
        initial_state,
        max_iterations,
        best_scores_vector):
    """
    Creating a set of information necessary to plot the results.
    :param initial_swarm: initial Swarm class instance
    :param final_swarm: final Swarm class instance
    :param chosen_states: states to be compared between orbits
    :param initial_state: the first point of the original orbit
    :param max_iterations: number of iterations
    :param best_scores_vector: vector of best scores in each iteration
    :return: set of data used by plotting functions
    """
    series1 = []
    series2 = []

    for source in initial_swarm.sources:
        series1.append(convert_to_metric_units(deepcopy(source.state)))

    for source in final_swarm.sources:
        series2.append(convert_to_metric_units(deepcopy(source.state)))

    solution = Food(final_swarm.global_best_state, final_swarm)
    solution.calculate_cost()
    time_vect = np.array(chosen_states['Time (TU)'])
    vect_size = len(time_vect)
    time_span = time_vect[vect_size - 1] + 0.01
    propagate = solution.propagate(time_vect, time_span)
    propagated_trajectory = propagate.y[:3] * 389703
    original_trajectory = np.array(chosen_states)[:, 1:4] * 389703

    return (original_trajectory, propagated_trajectory, initial_state[:3] * 389703,
            final_swarm.global_best_state[:3] * 389703, series1, series2)


def abc_alg(
        max_iterations,
        population_size,
        employee_phase_neighbours,
        onlooker_phase_neighbours,
        number_of_measurements,
        neighbours_pos_limits,
        neighbours_vel_limits,
        inactive_cycles_limit,
        inactive_cycles_setter=0,
        probability_distribution_setter=0,
        generating_method=0,
        neighbourhood_type=0,
        neigh_percent=0.02,
        dim_probability=0.5,
        orbit_filepath="../Orbits/L2_7days.txt"):
    """
    Heart of ABC algorithm.
    :param max_iterations: number of iterations
    :param population_size: number of particles
    :param employee_phase_neighbours: number of neighbours visited during employee phase
    :param onlooker_phase_neighbours: number of neighbours visited during onlooker phase
    :param number_of_measurements: number of points where the orbits are compared
    :param neighbours_pos_limits: defines the limits of neighbourhood for sources' positions
    :param neighbours_vel_limits: defines the limits of neighbourhood for sources' velocities
    :param inactive_cycles_limit: number of iterations where no improvement of a source is accepted
    :param inactive_cycles_setter: optional - modificates the number of inactive cycles
    depending on the current iteration
    :param probability_distribution_setter: optional - changes the probability the onlooker bees choose the source with
    :param generating_method: optional - defines the way new source is generated
    :param neighbourhood_type: optional - defines the way neighbourhood is handled
    :param neigh_percent: optional - defines the limits of velocity tolerance while generating new sources
    :param dim_probability: optional - defines the probability of modyfing a given dimention value
    :param orbit_filepath: path to raw data from NASA database
    :return:
    """
    file_path = orbit_filepath
    """
    Data loading.
    """
    time_vect, states, initial_state, initial_random, chosen_states = (
        transfer_raw_data_to_trajectory(
            file_path, population_size, number_of_measurements))

    swarm = Swarm(
        population_size,
        number_of_measurements,
        max_iterations,
        chosen_states,
        initial_state)
    swarm.generate_initial_population(initial_random)

    solution = Food(initial_state, swarm)
    time_vect = np.array(chosen_states['Time (TU)'])
    vect_size = len(time_vect)
    time_span = time_vect[vect_size - 1] + 1
    propagate = solution.propagate(time_vect, time_span)
    propagated_trajectory = propagate.y[:3]
    original_trajectory = np.array(propagated_trajectory)
    swarm.original_trajectory = original_trajectory

    for source in swarm.sources:
        source.calculate_cost()
    initial_swarm = deepcopy(swarm)
    best_scores_vector = []

    """
    ============================BEGINNING OF THE ALGORITHM'S ITERATIONS PROCEDURE===========================
    """
    for it in range(max_iterations):
        print("iter. no. ", it)

        """
        Employee phase
        Actions:
        1. Set all active_flag values to False (new cycle reset).
        2. Create new solutions as neighbours of each current food source.
        3. Choose the best one, based on costs (the lower, the better). Forget the others.
        """

        for source in swarm.sources:
            source.active_flag = False
            source.generate_neighbours(
                employee_phase_neighbours,
                neighbours_pos_limits,
                neighbours_vel_limits,
                neighbourhood_type,
                dim_probability)
            source.choose_best_neighbour()

        """
        Onlooker phase
        Actions:
        1. Calculate the probability of choosing the source: the better fit, the higher probability.
        2. Based on the mentioned probability, visit a source and try to find a better neighbour.
        """

        probabilities_vector = [0]

        if probability_distribution_setter == 0:
            score_list = [source.score for source in swarm.sources]
            min_cost = np.min(score_list)
            max_cost = np.max(score_list)
            range_score = min_cost + max_cost
            total_cost = np.sum(
                [range_score - source.score for source in swarm.sources])
            for source in swarm.sources:
                source.probability = (range_score - source.score) / total_cost
                probabilities_vector.append(
                    probabilities_vector[-1] + source.probability)

        elif probability_distribution_setter == 1:
            total_cost = np.sum([1 / source.score for source in swarm.sources])
            for source in swarm.sources:
                source.probability = (1 / source.score) / total_cost
                probabilities_vector.append(
                    probabilities_vector[-1] + source.probability)

        for _ in range(swarm.population_size):
            probability = random.uniform(0, 1)
            for idx, border_prob in enumerate(probabilities_vector, -1):
                if border_prob > probability:
                    chosen_state_idx = idx
                    break
            swarm.sources[chosen_state_idx].generate_neighbours(
                onlooker_phase_neighbours,
                neighbours_pos_limits,
                neighbours_vel_limits,
                neighbourhood_type,
                dim_probability)
            swarm.sources[chosen_state_idx].choose_best_neighbour()

        """
        Scout phase
        Actions:
        1. Establish the sources that weren't updated for a given number of iterations.
        2. Generate new sources in place of the found inactive ones.
        """

        if inactive_cycles_setter:
            if it >= 0:
                inactive_cycles_limit = 1
            if it >= 50:
                inactive_cycles_limit = 3
            if it >= 100:
                inactive_cycles_limit = 5
            if it >= 200:
                inactive_cycles_limit = 7

        for source in swarm.sources:
            if source.active_flag is False:
                source.inactive_cycles += 1  # Since all updates cause inactive_cycle drop by -1,
                # it will only increase when no update was done
                if source.inactive_cycles >= inactive_cycles_limit and it >= 1:
                    source.scout_new_source(
                        initial_state, generating_method, neigh_percent)

        """
        Score assessment phase
        Actions:
        1. Find the best score in this iteration.
        2. If it is better than the previous best, update it and save the state for which the score is achieved.
        """

        iteration_best_score_idx = np.argmin(
            [source.score for source in swarm.sources])
        iteration_best_state = swarm.sources[iteration_best_score_idx].state
        if swarm.sources[iteration_best_score_idx].score < swarm.global_best_score:
            swarm.global_best_score = swarm.sources[iteration_best_score_idx].score
            swarm.global_best_state = iteration_best_state

        print('global best score: ', swarm.global_best_score)

        best_scores_vector.append(swarm.global_best_score)
    """
    ==================================END OF ALGORITHM'S ITERATION PROCEDURE===================================
    """
    final_swarm = deepcopy(swarm)
    return [
        initial_swarm,
        final_swarm,
        chosen_states,
        initial_state,
        max_iterations,
        best_scores_vector]


def main():
    """
    A test instance - will not be run when algorithm used by GUI
    """
    population_size = 20
    number_of_measurements = 35
    max_iterations = 300
    employee_phase_neighbours = 4
    onlooker_phase_neighbours = 4
    inactive_cycles_limit = 5
    neighbours_pos_limits = 12.5
    neighbours_vel_limits = 0.00025

    abc_alg(
        max_iterations,
        population_size,
        employee_phase_neighbours,
        onlooker_phase_neighbours,
        number_of_measurements,
        neighbours_pos_limits,
        neighbours_vel_limits,
        inactive_cycles_limit,
        inactive_cycles_setter=0,
        probability_distribution_setter=0,
        generating_method=0,
        neighbourhood_type=0,
        neigh_percent=0.02,
        dim_probability=0.5)


if __name__ == '__main__':
    main()
