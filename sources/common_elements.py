"""
Defines parent class for particle and food source. Adds a few common functionalities
as solution_propagation() and personal dictionary structure.
"""
from copy import deepcopy
import numpy as np
from scipy import integrate
from numpy import linalg
from sources import data_load


class PropagatedElement:
    """
    Parent class for Particle and Food. It defines the common attributes and computing methods.
    """
    def __init__(self, swarm, state):
        self.mu = 0.01215058560962404
        self.swarm = swarm
        self.state = state
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
            self._diff_equation, [
                0, time_span], [
                x, y, z, vx, vy, vz], t_eval=time_vect, method='DOP853')
        return propagated_trajectory

    def _diff_equation(self, t, parameters): # pylint: disable=unused-argument
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

    def calculate_cost(self):
        """
        Calculates the cost of each trajectory as a sum of differences between
        the expected and propagated results in [number_of_measurements] points.
        """
        time_vect = np.array(self.swarm.chosen_states['Time (TU)'])
        vect_size = len(time_vect)
        time_span = time_vect[vect_size - 1] + 0.01
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


class Swarm:
    # pylint: disable=R0902, R0903, R0913, R0917
    """
    Creates a swarm that should be universal for all swarm algorithms.
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
        self.initial_state = initial_state
        self.elements = []
        self.global_best_score = np.inf
        self.global_best_state = []
        self.original_trajectory = None
        self.filepath = None

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


def solution_propagation(initial_state, swarm, chosen_states):
    """
    :param initial_state: the 6-dimensional vector of satellite's initial position and velocity
    :param swarm: Swarm class object where the propagated element belongs to
    :param chosen_states: array of satelite's positions and velocities in given moments of time
    """
    solution = PropagatedElement(swarm, initial_state)
    time_vect = np.array(chosen_states['Time (TU)'])
    vect_size = len(time_vect)
    time_span = time_vect[vect_size - 1] + 0.01
    propagate = solution.propagate(time_vect, time_span)
    propagated_trajectory = propagate.y[:3]
    original_trajectory = np.array(propagated_trajectory)
    swarm.original_trajectory = original_trajectory


def final_plot(
        initial_swarm,
        final_swarm,
        chosen_states,
        initial_state):
    """
    Creating a set of information necessary to plot the results.
    :param initial_swarm: initial Swarm class instance
    :param final_swarm: final Swarm class instance
    :param chosen_states: states to be compared between orbits
    :param initial_state: the first point of the original orbit
    :return: set of data used by plotting functions
    """
    series1 = []
    series2 = []

    for source in initial_swarm.elements:
        series1.append(data_load.convert_to_metric_units(deepcopy(source.state)))

    for source in final_swarm.elements:
        series2.append(data_load.convert_to_metric_units(deepcopy(source.state)))

    solution = PropagatedElement(final_swarm, final_swarm.global_best_state)
    solution.calculate_cost()
    time_vect = np.array(chosen_states['Time (TU)'])
    vect_size = len(time_vect)
    time_span = time_vect[vect_size - 1] + 0.01
    propagate = solution.propagate(time_vect, time_span)
    propagated_trajectory = propagate.y[:3] * 389703
    original_trajectory = np.array(chosen_states)[:, 1:4] * 389703

    return (original_trajectory, propagated_trajectory, initial_state[:3] * 389703,
            final_swarm.global_best_state[:3] * 389703, series1, series2)
