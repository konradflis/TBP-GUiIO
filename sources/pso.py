import random
from copy import deepcopy
from scipy import integrate
import numpy as np
from numpy import linalg
from Sources.data_load import transfer_raw_data_to_trajectory, convert_to_metric_units
from Sources.plot_functions import present_results


class Particle:
    def __init__(self, state, swarm):
        self.state = state
        self.swarm = swarm
        self.mu = 0.01215058560962404
        self.inertia = self.swarm.inertia
        self.c1 = self.swarm.c1
        self.c2 = self.swarm.c2
        self.particle_best_state = self.state
        self.particle_current_score = np.inf
        self.particle_best_score = np.inf
        self.particle_velocity = np.zeros(6)

    def update_particle(self):
        """
        Method updating particle's position and velocity (state)
        based on propagation results and meta-parameters
        :return: None
        """
        r1 = random.random()
        r2 = random.random()
        self.particle_velocity = (self.inertia *
                                  self.particle_velocity +
                                  self.c1 *
                                  r1 *
                                  (self.particle_best_state -
                                   self.state) +
                                  self.c2 *
                                  r2 *
                                  (self.swarm.global_best_state -
                                   self.state))

        self.state = self.state + self.particle_velocity
        self.inertia = self.swarm.inertia
        self.c1 = self.swarm.c1
        self.c2 = self.swarm.c2

    def propagate(self, time_vect, time_span):
        """
        Method propagating the initial conditions using the defined differential equations.
        :param time_vect: points in time that are used
        to compare the original and propagated trajectory
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

    def calculate_cost(self):
        """
        Calculates the cost of each trajectory as a sum of differences between
        the expected and propagated results in
        :number_of_measurements: points.
        :return: None
        """
        time_vect = np.array(self.swarm.chosen_states['Time (TU)'])
        vect_size = len(time_vect)
        time_span = time_vect[vect_size - 1] + 0.01
        propagate = self.propagate(time_vect, time_span)
        original_trajectory = self.swarm.original_trajectory.T
        propagated_trajectory = propagate.y

        error = 0
        if propagated_trajectory.shape[1] == self.swarm.number_of_measurement:
            for it in range(self.swarm.number_of_measurement):
                error += linalg.norm(original_trajectory[it,
                                     :3] - propagated_trajectory[:3, it])
            self.particle_current_score = error
        else:
            self.particle_current_score = np.inf

    def check_if_best(self):
        """
        Updates the particle's best score and best state if a new best score was found.
        :return: None
        """
        if self.particle_current_score < self.particle_best_score:
            self.particle_best_score = self.particle_current_score
            self.particle_best_state = self.state

    def convert_to_metric_units(self):
        """
        Converts the LU and LU/TU units to km and km/s respectively
        :return: None
        """
        lu_to_km_coeff = 389703
        tu_to_s_coeff = 382981
        self.state[:3] = self.state[:3] * lu_to_km_coeff
        self.state[3:] = self.state[3:] * (lu_to_km_coeff / tu_to_s_coeff)
        self.particle_best_state[:3] = self.particle_best_state[:3] * lu_to_km_coeff
        self.particle_best_state[3:] = (self.particle_best_state[3:] *
                                        (lu_to_km_coeff / tu_to_s_coeff))


class Swarm:
    def __init__(
            self,
            max_iterations,
            population_size,
            number_of_measurements,
            chosen_states,
            global_best_state,
            inertia,
            c1,
            c2):
        self.inertia = inertia
        self.c1 = c1
        self.c2 = c2
        self.number_of_measurement = number_of_measurements
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.global_best_score = np.inf
        self.global_best_state = global_best_state
        self.chosen_states = chosen_states
        self.particles = []
        self.filepath = None

    def generate_initial_population(
            self,
            initial_random,
            opt_multistart=0,
            opt_number_of_starts=20,
            opt_best_velocity=None):
        """
        Generates a new population of Particle class objects,
        based on the initial points that are passed as an argument.
        :param initial_random: list of 6-dimensional points representing
         each particle's initial position and velocity
        :param opt_multistart: if 1, multistart modificaiton is active
        :param opt_number_of_starts: if multistart active, create the swarm with 20 initial starts
        :return: None
        """
        if opt_best_velocity is None:
            opt_best_velocity = [
                0,
                0,
                0]
        if not opt_multistart:
            for idx in range(self.population_size):
                initial_conditions = initial_random[idx]
                particle_object = Particle(initial_conditions, self)
                self.particles.append(particle_object)

        if opt_multistart:
            self.particles = multistart(
                opt_number_of_starts,
                1,
                self.population_size,
                35,
                self.filepath,
                self.inertia,
                self.c1,
                self.c2,
                opt_best_velocity,
                self)

    def update_global_best(self):
        """
        Updates the global best score and state using the best particle's properties.
        :return: None
        """
        scores = [particle.particle_current_score for particle in self.particles]
        min_val = np.min(scores)
        min_idx = scores.index(min(scores))
        if min_val < self.global_best_score:
            self.global_best_score = min_val
            self.global_best_state = self.particles[min_idx].state

    def inertia_setter(
            self,
            it,
            opt_setter=0,
            opt_starting_inertia=1,
            opt_stop_inertia=1.0):
        """
        If active, it modifies the inertia during the algorithm so that the velocity changes
        are adapted to the conditions.
        :param it: current iteration
        :param opt_setter: chooses the way of inertia's modification
        :param opt_starting_inertia: define the initial inertia value
        :param opt_stop_inertia: define the final inertia value
        :return: None
        """

        if not opt_setter:
            pass

        if opt_setter == 1:
            step = (opt_starting_inertia - opt_stop_inertia) / \
                   self.max_iterations
            self.inertia = self.inertia - step

        if opt_setter == 2:
            if it >= 0:
                self.inertia = 0.95
            if it > 20:
                self.inertia = 0.9
            if it > 50:
                self.inertia = 0.85
            if it > 100:
                self.inertia = 0.8
            if it > 150:
                self.inertia = 0.75

        if opt_setter == 3:
            if it > 0:
                self.inertia = 0.95
            if it > 10:
                self.inertia = 0.9
            if it > 50:
                self.inertia = 0.825
            if it > 100:
                self.inertia = 0.75
            if it > 150:
                self.inertia = 0.65

    def n_setter(self, it, opt_setter=0):
        """
        :param it: iteration number
        :param opt_setter: indicates the way social and cognitive coeffs change
        """
        if not opt_setter:
            pass

        if opt_setter == 1:
            if it >= 0:
                self.c1 = 1
                self.c2 = 1
            if it > 20:
                self.c1 = 1.2
                self.c2 = 0.8
            if it > 50:
                self.c1 = 1.4
                self.c2 = 0.6
            if it > 100:
                self.c1 = 1.6
                self.c2 = 0.4
            if it > 150:
                self.c1 = 1.75
                self.c2 = 0.25

        if opt_setter == 2:
            if it >= 0:
                self.c1 = 1
                self.c2 = 1
            if it > 20:
                self.c1 = 0.8
                self.c2 = 1.2
            if it > 50:
                self.c1 = 0.6
                self.c2 = 1.4
            if it > 100:
                self.c1 = 0.4
                self.c2 = 1.6
            if it > 150:
                self.c1 = 0.25
                self.c2 = 1.75

    def convert_to_metric_units(self):
        """
        Converts the LU and LU/TU units to km and km/s respectively
        :return: None
        """
        lu_to_km_coeff = 389703
        tu_to_s_coeff = 382981
        self.global_best_state[:3] = self.global_best_state[:3] * lu_to_km_coeff
        self.global_best_state[3:] = self.global_best_state[3:] * (lu_to_km_coeff / tu_to_s_coeff)


def multistart(
        number_of_starts,
        max_iterations,
        population_size,
        number_of_measurements,
        file_path,
        inertia,
        c1,
        c2,
        best_velocity,
        sum_swarm):
    """
    Generates a new population based on a few initial iterations
    that provide with the best particles of each one.
    Parameters typical to PSO algorithm and used in all functions.
    :return: initial_population
    """
    print('===MULTISTART===')
    initial_population = []
    number_of_top_elems = population_size // number_of_starts

    for _ in range(number_of_starts):
        print("starting iteration nr", _)
        time_vect, states, initial_state, initial_random, chosen_states = (
            transfer_raw_data_to_trajectory(file_path, population_size, number_of_measurements,
                                            opt_part2=1,
                                            opt_best_velocity=best_velocity))

        swarm = Swarm(
            max_iterations,
            population_size,
            number_of_measurements,
            chosen_states,
            np.zeros(6),
            inertia,
            c1,
            c2)
        swarm.generate_initial_population(initial_random, opt_multistart=0)

        solution = Particle(initial_state, swarm)
        time_vect = np.array(chosen_states['Time (TU)'])
        vect_size = len(time_vect)
        time_span = time_vect[vect_size - 1] + 1
        propagate = solution.propagate(time_vect, time_span)
        propagated_trajectory = propagate.y[:3]
        original_trajectory = np.array(propagated_trajectory)
        swarm.original_trajectory = original_trajectory

        swarm.update_global_best()
        for it in range(swarm.max_iterations):
            for particle in swarm.particles:
                particle.calculate_cost()
                particle.check_if_best()
            swarm.update_global_best()
            for particle in swarm.particles:
                particle.update_particle()
        scores = [
            particle.particle_current_score for particle in swarm.particles]
        scores_idx = sorted(
            range(
                len(scores)),
            key=lambda i: scores[i])[
                     :number_of_top_elems]
        for idx in scores_idx:
            initial_population.append(
                Particle(
                    swarm.particles[idx].state,
                    sum_swarm))
    return initial_population


def final_plot(
        initial_swarm,
        final_swarm,
        chosen_states,
        initial_state,
        max_iterations,
        best_scores_vector):
    """
    Creates a set of information necessary to plot the results.
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

    for particle in initial_swarm.particles:
        series1.append(convert_to_metric_units(deepcopy(particle.state)))

    for particle in final_swarm.particles:
        series2.append(convert_to_metric_units(deepcopy(particle.state)))

    solution = Particle(final_swarm.global_best_state, final_swarm)
    solution.calculate_cost()
    time_vect = np.array(chosen_states['Time (TU)'])
    vect_size = len(time_vect)
    time_span = time_vect[vect_size - 1] + 0.01
    propagate = solution.propagate(time_vect, time_span)
    propagated_trajectory = propagate.y[:3] * 389703
    original_trajectory = np.array(chosen_states)[:, 1:4] * 389703
    present_results(initial_state, final_swarm.global_best_state)

    return (original_trajectory, propagated_trajectory, initial_state[:3] * 389703,
            final_swarm.global_best_state[:3] * 389703, series1, series2)


def pso(
        max_iterations,
        population_size,
        number_of_measurements,
        inertia,
        c1,
        c2,
        if_inertia_adaptation=0,
        stop_inertia=0.6,
        if_n_setter=0,
        if_best_velocity=0,
        opt_best_velocity=None,
        if_multistart=0,
        number_of_multistarts=20,
        orbit_filepath="../Orbits/L2_7days.txt"):
    """
    Heart of the PSO algorithm.
    :param max_iterations: number of iterations
    :param population_size: number of particles
    :param number_of_measurements: number of points where the orbits are compared
    :param inertia: value of inertia
    :param c1: value of self-trust coeff
    :param c2: value of social coeff
    :param if_inertia_adaptation: optional - modificates the inertia values based on iteration
    :param stop_inertia: optional - the final inertia achieved during the last iteration
    :param if_n_setter: optional - modificates the cognitive coeffs
    :param if_best_velocity: optional - indicates if the best found velocity so far
    should impact further algorithm's instances
    :param opt_best_velocity: optional - if best velocity modification is ON,
    it provides with the value
    :param if_multistart: optional - multistart modification
    :param number_of_multistarts: optional - if multistart modification is ON, it indicates
    how many different initial swarms it should generate
    :param orbit_filepath: path raw data from NASA database
    :return:
    """
    if opt_best_velocity is None:
        opt_best_velocity = [
            0,
            0,
            0]
    file_path = orbit_filepath
    time_vect, states, initial_state, initial_random, chosen_states = \
        (transfer_raw_data_to_trajectory(
            file_path, population_size, number_of_measurements, opt_part2=if_best_velocity,
            opt_best_velocity=opt_best_velocity))
    swarm = Swarm(
        max_iterations,
        population_size,
        number_of_measurements,
        chosen_states,
        np.zeros(6),
        inertia,
        c1,
        c2)
    swarm.filepath = file_path
    swarm.generate_initial_population(
        initial_random,
        opt_multistart=if_multistart,
        opt_number_of_starts=number_of_multistarts,
        opt_best_velocity=opt_best_velocity)
    solution = Particle(initial_state, swarm)
    time_vect = np.array(chosen_states['Time (TU)'])
    vect_size = len(time_vect)
    time_span = time_vect[vect_size - 1] + 0.01
    propagate = solution.propagate(time_vect, time_span)
    propagated_trajectory = propagate.y[:3]
    original_trajectory = np.array(propagated_trajectory)
    swarm.original_trajectory = original_trajectory

    initial_swarm = deepcopy(swarm)
    best_scores_vector = []
    swarm.update_global_best()

    for it in range(max_iterations):
        print("iter. no", it)

        for particle in swarm.particles:
            particle.calculate_cost()
            particle.check_if_best()
        swarm.update_global_best()
        swarm.inertia_setter(
            it,
            opt_setter=if_inertia_adaptation,
            opt_starting_inertia=inertia,
            opt_stop_inertia=stop_inertia)
        swarm.n_setter(it, opt_setter=if_n_setter)
        for particle in swarm.particles:
            particle.update_particle()
        print(swarm.global_best_score)
        best_scores_vector.append(swarm.global_best_score)

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
    max_iterations = 2
    population_size = 50
    number_of_measurements = 35
    inertia = 0.95
    c1 = 1.5
    c2 = 0.5
    pso(max_iterations, population_size, number_of_measurements, inertia, c1, c2)


if __name__ == '__main__':
    main()
