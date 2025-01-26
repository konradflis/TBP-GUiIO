"""
Particle Swarm Optimization implementation.
"""
import random
from copy import deepcopy
import numpy as np
from sources.data_structures import MandatorySettingsPSO, OptionalSettingsPSO
from sources.common_elements import PropagatedElement, ModelProperties, Swarm


class Particle(PropagatedElement):
    """
    Class defining a particle - the fundamental concept of Particle Swarm Optimization.
    """

    def __init__(self, state, swarm, model):
        super().__init__(state, model)
        self.swarm = swarm
        self.inertia = self.swarm.inertia
        self.c1 = self.swarm.c1
        self.c2 = self.swarm.c2
        self.best_state = self.state
        self.best_score = np.inf
        self.velocity = np.zeros(6)

    def update(self):
        """
        Method updating particle's position and velocity (state)
        based on propagation results and meta-parameters
        """
        r1 = random.random()
        r2 = random.random()
        self.velocity = (self.inertia *
                         self.velocity +
                         self.c1 *
                         r1 *
                         (self.best_state -
                          self.state) +
                         self.c2 *
                         r2 *
                         (self.swarm.global_best_state -
                          self.state))

        self.state = self.state + self.velocity
        self.inertia = self.swarm.inertia
        self.c1 = self.swarm.c1
        self.c2 = self.swarm.c2

    def check_if_best(self):
        """
        Updates the particle's best score and best state if a new best score was found.
        """
        if self.score < self.best_score:
            self.best_score = self.score
            self.best_state = self.state

    def convert_to_metric_units(self):
        """
        Converts the LU and LU/TU units to km and km/s respectively
        """
        lu_to_km_coeff = 389703
        tu_to_s_coeff = 382981
        self.state[:3] = self.state[:3] * lu_to_km_coeff
        self.state[3:] = self.state[3:] * (lu_to_km_coeff / tu_to_s_coeff)
        self.best_state[:3] = self.best_state[:3] * lu_to_km_coeff
        self.best_state[3:] = (self.best_state[3:] *
                               (lu_to_km_coeff / tu_to_s_coeff))


class SwarmPSO(Swarm):
    # pylint: disable=R0902, R0903, R0913, R0917
    """
    Class defining a swarm of particles that explore and exploit the space.
    """

    def __init__(self,
                 max_iterations,
                 population_size,
                 model,
                 inertia,
                 c1,
                 c2):
        super().__init__(population_size,
                         max_iterations,
                         model
                         )
        self.inertia = inertia
        self.c1 = c1
        self.c2 = c2

    def generate_initial_population(
            self,
            opt_multistart=0,
            opt_number_of_starts=20,
            opt_if_two_stage_pso=0,
            opt_best_velocity=None):
        """
        Generates a new population of Particle class objects,
        based on the initial points that are passed as an argument.
        :param opt_multistart: if 1, multistart modification is active
        :param opt_number_of_starts: if multistart active, create the swarm with 20 initial starts
        :param opt_if_two_stage_pso: if 1, PSO2 algorithm is used
        :param opt_best_velocity: if 1, best velocity from previous iterations has impact on
        new particles (used in PSO2)
        """
        initial_random = super().generate_initial_population(opt_if_two_stage_pso, opt_best_velocity)

        if not opt_multistart:
            for idx in range(self.population_size):
                initial_conditions = initial_random[idx]
                particle_object = Particle(initial_conditions, self, self.model)
                self.elements.append(particle_object)

        if opt_multistart:
            mandatory = MandatorySettingsPSO(
                max_iterations=1,
                population_size=self.population_size,
                number_of_measurements=35,
                inertia=self.inertia,
                c1=self.c1,
                c2=self.c2
            )
            optional = OptionalSettingsPSO(
                number_of_multistarts=opt_number_of_starts
            )
            self.elements = multistart(
                self,
                mandatory,
                optional)


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
        """
        if not opt_setter:
            pass

        if opt_setter == 1:
            step = (opt_starting_inertia - opt_stop_inertia) / self.max_iterations
            self.inertia = self.inertia - step

        if opt_setter == 2:
            inertia_values = [(0, 0.95), (20, 0.9), (50, 0.85), (100, 0.8), (150, 0.75)]
            for threshold, inertia_value in inertia_values:
                if it > threshold:
                    self.inertia = inertia_value

        if opt_setter == 3:
            inertia_values = [(0, 0.95), (10, 0.9), (50, 0.825), (100, 0.75), (150, 0.65)]
            for threshold, inertia_value in inertia_values:
                if it > threshold:
                    self.inertia = inertia_value

    def c_setter(self, it, opt_setter=0):
        """
        :param it: iteration number
        :param opt_setter: indicates the way social and cognitive coeffs change
        """
        if not opt_setter:
            pass

        if opt_setter == 1:
            c_values = [
                (0, 1, 1),
                (20, 1.2, 0.8),
                (50, 1.4, 0.6),
                (100, 1.6, 0.4),
                (150, 1.75, 0.25)
            ]
            for threshold, c1_value, c2_value in c_values:
                if it >= threshold:
                    self.c1 = c1_value
                    self.c2 = c2_value

        if opt_setter == 2:
            c_values = [
                (0, 1, 1),
                (20, 0.8, 1.2),
                (50, 0.6, 1.4),
                (100, 0.4, 1.6),
                (150, 0.25, 1.75)
            ]
            for threshold, c1_value, c2_value in c_values:
                if it >= threshold:
                    self.c1 = c1_value
                    self.c2 = c2_value


def multistart(
        sum_swarm,
        mandatory,
        optional=None):
    """
    Generates a new population based on a few initial iterations
    that provide with the best particles of each one.
    Parameters typical to PSO algorithm and used in all functions.
    :return: initial_population - set of initial particles
    """
    if optional is None:
        optional = OptionalSettingsPSO()

    if optional.best_velocity is None:
        optional.best_velocity = [0, 0, 0]

    print('===MULTISTART===')
    initial_population = []

    for _ in range(optional.number_of_multistarts):
        print("===STARTING NEW SWARM===")
        model = ModelProperties(optional.orbit_filepath, mandatory.number_of_measurements)

        swarm = SwarmPSO(
            mandatory.max_iterations,
            mandatory.population_size,
            model,
            mandatory.inertia,
            mandatory.c1,
            mandatory.c2)

        swarm.generate_initial_population()

        swarm.update_global_best()
        for _ in range(swarm.max_iterations):
            for particle in swarm.elements:
                particle.calculate_cost()
                particle.check_if_best()
            swarm.update_global_best()
            for particle in swarm.elements:
                particle.update()
            print(swarm.global_best_score)
        scores = [
            particle.score for particle in swarm.elements]
        #Creates a list in form of (index, score), sorts it and choses 1/n scores in each of n runs
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1])
        scores_idx = [index for index, _ in indexed_scores[
                                            :(mandatory.population_size // optional.number_of_multistarts)]]
        for idx in scores_idx:
            initial_population.append(
                Particle(
                    swarm.elements[idx].state,
                    sum_swarm,
                    model))
    return initial_population


def pso(
        mandatory,
        optional=None):
    """
    Heart of the PSO algorithm.
    :param mandatory: structure of mandatory parameters, including:
    :> max_iterations: number of iterations
    :> population_size: number of particles
    :> number_of_measurements: number of points where the orbits are compared
    :> inertia: value of inertia
    :> c1: value of self-trust coeff
    :> c2: value of social coeff
    :param optional: structure of optional parameters, including:
    :> inertia_setter: optional - modificates the inertia values based on iteration
    :> stop_inertia: optional - the final inertia achieved during the last iteration
    :> c_setter: optional - modificates the cognitive coeffs
    :> if_best_velocity: optional - indicates if the best found velocity so far
    should impact further algorithm's instances
    :> best_velocity: optional - if best velocity modification is ON,
    it provides with the value
    :> multistart: optional - multistart modification
    :> number_of_multistarts: optional - if multistart modification is ON, it indicates
    how many different initial swarms it should generate
    :> orbit_filepath: path raw data from NASA database
    :return: set of information about the initial and final swarm
    """

    if optional is None:
        optional = OptionalSettingsPSO()

    if optional.best_velocity is None:
        optional.best_velocity = [0, 0, 0]

    model = ModelProperties(optional.orbit_filepath, mandatory.number_of_measurements)

    swarm = SwarmPSO(
        mandatory.max_iterations,
        mandatory.population_size,
        model,
        mandatory.inertia,
        mandatory.c1,
        mandatory.c2)

    swarm.generate_initial_population(
        opt_multistart=optional.multistart,
        opt_number_of_starts=optional.number_of_multistarts,
        opt_if_two_stage_pso=optional.if_best_velocity,
        opt_best_velocity=optional.best_velocity)

    initial_swarm = deepcopy(swarm)
    best_scores_vector = []
    swarm.update_global_best()

    for it in range(mandatory.max_iterations):
        print('iteration nr', it)
        for particle in swarm.elements:
            particle.calculate_cost()
            particle.check_if_best()
        swarm.update_global_best()
        swarm.inertia_setter(
            it,
            opt_setter=optional.inertia_setter,
            opt_starting_inertia=mandatory.inertia,
            opt_stop_inertia=optional.stop_inertia)
        swarm.c_setter(it, opt_setter=optional.c_setter)
        for particle in swarm.elements:
            particle.update()
        print(swarm.global_best_score)
        best_scores_vector.append(swarm.global_best_score)

    final_swarm = deepcopy(swarm)
    # pylint: disable=R0801
    return [
        initial_swarm,
        final_swarm,
        model.chosen_states,
        model.initial_state,
        mandatory.max_iterations,
        best_scores_vector]

def main():
    """
    A test instance - will not be run when algorithm used by GUI
    """
    pso(MandatorySettingsPSO(), OptionalSettingsPSO(multistart=1, number_of_multistarts=5))

if __name__ == '__main__':
    main()
