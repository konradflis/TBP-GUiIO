"""
Defines parent class for particle and food source, for their swarms and a general class for
the whole model - containing the information about the orbit and measurements.
"""
from copy import deepcopy
import numpy as np
from scipy import integrate
from numpy import linalg
from sources import data_load


class ModelProperties:
    """
    Class defining the three-body problem model properties - constants, orbits and other parameters.
    It is used by both Swarm and Particle classes to access the model-based information, like
    number of measurements of orbit, its initial state and original trajectory, period etc.
    Model equations are defined here as well.
    """
    # pylint: disable=R0902, R0903, R0913, R0917
    def __init__(self,
                 filepath,
                 number_of_measurements):
        self.mu = 0.01215058560962404
        self.filepath = filepath
        self.number_of_measurements = number_of_measurements
        self.period, self.all_states, self.initial_state, self.chosen_states = (
            data_load.transfer_raw_data_to_trajectory(self.filepath, self.number_of_measurements))
        self.original_trajectory = self.initial_solution_propagation()

    def diff_equation(self, t, parameters): # pylint: disable=unused-argument
        """
        :param t: function handler parameter
        :param parameters: positions and velocities (6 parameters altogether)
        :return: defined outputs of differential equations (positions and velocities)
        """
        x1, y1, z1, x2, y2, z2 = parameters
        r_se = np.sqrt(((x1 + self.mu) ** 2 + y1 ** 2 + z1 ** 2) ** 3)
        r_sm = np.sqrt(((x1 + self.mu - 1) ** 2 + y1 ** 2 + z1 ** 2) ** 3)

        _, out_x2 = np.array([x2, 2 * y2 + x1 - (1 - self.mu) *
                                   (x1 + self.mu) / r_se - self.mu * (x1 + self.mu - 1) / r_sm])
        _, out_y2 = np.array(
            [y2, -2 * x2 + y1 - (1 - self.mu) * y1 / r_se - self.mu * y1 / r_sm])
        _, out_z2 = np.array(
            [z2, -1 * (1 - self.mu) * z1 / r_se - self.mu * z1 / r_sm])

        return [x2, y2, z2, out_x2, out_y2, out_z2]

    def refresh_model(self):
        """
        If needed, model can be refreshed - based on a new file
        with new data or number of measurements.
        """
        self.period, self.all_states, self.initial_state, self.chosen_states = (
            data_load.transfer_raw_data_to_trajectory(self.filepath, self.number_of_measurements))

    def initial_solution_propagation(self):
        """
        Finding the original trajectory by propagating the initial state.
        :return: original trajectory (a set of points with
        positions and velocities in given moments)
        """
        solution = PropagatedElement(self.initial_state, self)
        time_vect = np.array(self.chosen_states['Time (TU)'])
        time_span = time_vect[-1] + 0.01
        propagate = solution.propagate(time_vect, time_span)
        propagated_trajectory = propagate.y[:3]
        original_trajectory = np.array(propagated_trajectory)
        return original_trajectory


class PropagatedElement:
    """
    Parent class for Particle and Food. It defines the common attributes and computing methods.
    """
    def __init__(self, state, model):
        self.state = state
        self.model = model
        self.score = np.inf

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
            self.model.diff_equation, [
                0, time_span], [
                x, y, z, vx, vy, vz], t_eval=time_vect, method='DOP853')
        return propagated_trajectory

    def calculate_cost(self):
        """
        Calculates the cost of each trajectory as a sum of differences between
        the expected and propagated results in [number_of_measurements] points.
        """
        time_vect = np.array(self.model.chosen_states['Time (TU)'])
        vect_size = len(time_vect)
        time_span = time_vect[vect_size - 1] + 0.01
        propagate = self.propagate(time_vect, time_span)
        original_trajectory = self.model.original_trajectory.T
        propagated_trajectory = propagate.y

        error = 0
        if propagated_trajectory.shape[1] == self.model.number_of_measurements:
            for it in range(self.model.number_of_measurements):
                error += linalg.norm(original_trajectory[it,
                                     :3] - propagated_trajectory[:3, it])
            self.score = error
        else:
            self.score = np.inf


class Swarm:
    # pylint: disable=R0902, R0903, R0913, R0917
    """
    Creates a swarm that should be universal for all swarm algorithms.
    Defines its basic characteristics.
    """

    def __init__(
            self,
            population_size,
            max_iterations,
            model):
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.elements = []
        self.global_best_score = np.inf
        self.global_best_state = []
        self.model = model

    def generate_initial_population(self,
            opt_if_two_stage_pso=0,
            opt_best_velocity=None):
        """
        Creates an initial population of swarm elements.
        :param opt_if_two_stage_pso: optional: defines the way
        points are generated in the search space
        :param opt_best_velocity: optional: defines if best
        found velocity so far should impact
        the new set of initial particles
        """
        # pylint: disable=R0914
        lu_to_km_coeff = 389703
        tu_to_s_coeff = 382981
        pos_limits = 250 / lu_to_km_coeff
        vel_limits = 0.1 / (lu_to_km_coeff / tu_to_s_coeff)
        initial_random = []

        if not opt_if_two_stage_pso:
            for _ in range(self.population_size):
                position = [np.random.uniform(-pos_limits, pos_limits) for _ in range(3)]
                velocity = [np.random.uniform(-vel_limits, vel_limits) for _ in range(3)]
                random_vect = position + velocity
                initial_random.append(self.model.initial_state + random_vect)
        else:
            for _ in range(self.population_size):
                position = [np.random.uniform(-pos_limits, pos_limits) for _ in range(3)]
                velocity = [0, 0, 0]
                random_vect = position + velocity
                initial_random.append(self.model.initial_state + random_vect)

            for elem in initial_random:
                percent = 0.001
                elem[3:] = [opt_best_velocity[0] + np.random.uniform(-1 * percent, percent),
                            opt_best_velocity[1] + np.random.uniform(-1 * percent, percent),
                            opt_best_velocity[2] + np.random.uniform(-1 * percent, percent)]
        return initial_random


    def update_global_best(self):
        """
        Finds the global best solution (its score and state) and updates it.
        """
        iteration_best_score_idx = np.argmin(
            [source.score for source in self.elements])
        iteration_best_state = self.elements[iteration_best_score_idx].state
        if self.elements[iteration_best_score_idx].score < self.global_best_score:
            self.global_best_score = self.elements[iteration_best_score_idx].score
            self.global_best_state = iteration_best_state

    def convert_to_metric_units(self):
        """
        Converts the LU and LU/TU units to km and km/s respectively
        """
        lu_to_km_coeff = 389703
        tu_to_s_coeff = 382981
        self.global_best_state[:3] = self.global_best_state[:3] * lu_to_km_coeff
        self.global_best_state[3:] = self.global_best_state[3:] * (lu_to_km_coeff / tu_to_s_coeff)


def final_plot(
        initial_swarm,
        final_swarm,
        chosen_states,
        initial_state,
        settings):
    # pylint: disable=R0902, R0903, R0913, R0917
    """
    Creating a set of information necessary to plot the results.
    :param initial_swarm: initial Swarm class instance
    :param final_swarm: final Swarm class instance
    :param chosen_states: states to be compared between orbits
    :param initial_state: the first point of the original orbit
    :param settings: the way plot will be generated
    :settings.density:
    :> 0: default: only the measurement point
    :> 1: dense: seemingly-continuous orbits
    :settings.period: time span of the plot, multiple of one period
    :return: set of data used by plotting functions
    """
    series1 = []
    series2 = []

    for source in initial_swarm.elements:
        series1.append(data_load.convert_to_metric_units(deepcopy(source.state)))

    for source in final_swarm.elements:
        series2.append(data_load.convert_to_metric_units(deepcopy(source.state)))

    found_solution = PropagatedElement(final_swarm.global_best_state, final_swarm.model)
    global_best_solution = PropagatedElement(initial_state, initial_swarm.model)

    original_trajectory = []
    propagated_trajectory = []

    if settings.density == 0:
        time_vect = np.array(chosen_states['Time (TU)'])
        time_span = time_vect[-1] + 0.01
        original_trajectory = data_load.convert_to_metric_units(
            deepcopy(initial_swarm.model.original_trajectory))
        propagated_trajectory = data_load.convert_to_metric_units(
            found_solution.propagate(time_vect, time_span).y[:3])

    if settings.density == 1:
        time_span = initial_swarm.model.period
        time_vect = np.linspace(0, time_span, 500)
        original_propagate = global_best_solution.propagate(time_vect, time_span)
        original_trajectory = data_load.convert_to_metric_units(original_propagate.y[:3])
        time_vect = np.linspace(0, float(
            time_span * settings.periods), int(500 * settings.periods))
        propagated_trajectory = data_load.convert_to_metric_units(
            found_solution.propagate(time_vect, time_span * settings.periods).y[:3])

    return (original_trajectory, propagated_trajectory, initial_state[:3] * 389703,
            final_swarm.global_best_state[:3] * 389703, series1, series2)
