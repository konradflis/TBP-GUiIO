"""
Firefly Algorithm implementation for orbital trajectory optimization.

Based on Xin-She Yang's Firefly Algorithm (2009) & adaptations for 3 body problem
"""

from copy import deepcopy
import numpy as np
from sources.common_elements import PropagatedElement, ModelProperties, Swarm
from sources.data_structures import MandatorySettingsFA, OptionalSettingsFA


class SwarmFirefly(Swarm):
    """
    Creates a swarm of fireflies (elements).
    Defines its basic characteristics.
    """

    def generate_initial_population(self,
                                    opt_if_two_stage_pso=0,
                                    opt_best_velocity=None):
        """
        Generates initial population of fireflies, based on the initial points
        that are passed as an argument.
        """
        initial_random = super().generate_initial_population()

        for idx in range(self.population_size):
            initial_conditions = initial_random[idx]
            firefly = Firefly(initial_conditions, self, self.model)
            self.elements.append(firefly)

class Firefly(PropagatedElement):
    """
    Defines a firefly and all its properties
    """

    def __init__(self, state, swarm, model):
        super().__init__(state, model)
        self.swarm = swarm
        self.brightness = None
        self.update_brightness()

    def update_brightness(self):
        """
        Update brightness based on calculateed cost
        """
        # cost calc
        self.calculate_cost()
        self.brightness = 1 / (self.score + 1e-10) # 1e-10 to escape from division by zero

    def move_towards(self, other, alpha, gamma, beta0):
        """
        Firefly movement with auto-matched randomization ranges
        using the same limits as in Swarm's initialization
        """
        # Calculate distance and attractiveness
        r = np.linalg.norm(self.state - other.state) # cartesian distance
        beta = beta0 * np.exp(-gamma * r ** 2) # attractiveness function - exponential
        # beta can also be defined as β = beta0 / (1 + gamma * r**2) - therefore beta
        # decreases monotonically - inversed squared


        # Generate random vector in with support of swarm (take first row to get vector)
        temp_swarm = Swarm(1, 1, self.model)
        temp_population = temp_swarm.generate_initial_population()
        random_vector = temp_population[0] - self.model.initial_state

        # Update position
        # XIN-SHE YANG proposed 3 types of
        new_state = self.state + beta * (other.state - self.state) + alpha * random_vector
        # movement of fireflies

        # Apply bounds
        clipped_state = temp_swarm.generate_initial_population()[0]
        self.state = new_state + (clipped_state - temp_population[0])

        self.update_brightness()

def firefly_alg(mandatory, optional=None):
    """
    Heart of Firefly Algorithm implementation.

    :param mandatory: structure of mandatory parameters, including:
    :> max_iterations: number of iterations
    :> population_size: number of fireflies
    :> number_of_measurements: number of points where the orbits are compared
    :> neighbours_pos_limits: defines the limits of neighbourhood for positions
    :> neighbours_vel_limits: defines the limits of neighbourhood for velocities
    :> inactive_cycles_limit: number of iterations without improvement before restart

    :param optional: structure of optional parameters, including:
    :> alpha_decay: exponential decay rate for randomization parameter
    :> orbit_filepath: path to raw data from NASA database

        NOT IMPLEMENTED OPTIONAL
    :> attractiveness_function: attractiveness function type
    :> distance_metric: distance calculation method
    :> randomization_type: type of randomization
    :> bounds: search space bounds for each dimension

    :return: list of information about the initial and final solutions

    Note: Firefly-specific behavior:
    - Movement based on brightness attractiveness
    - Exponential decay of randomization (alpha)
    - Automatic firefly restart when inactive
    """

    if optional is None:
        optional = OptionalSettingsFA()

    # orbital mode init
    model = ModelProperties(optional.orbit_filepath, mandatory.number_of_measurements)
    # swarm init
    swarm = SwarmFirefly(
        mandatory.population_size,
        mandatory.max_iterations,
        model
    )
    swarm.generate_initial_population()
    # Calculate cost for first swarm randomly initialized
    for firefly in swarm.elements:
        firefly.calculate_cost()
    swarm.update_global_best()
    initial_swarm = deepcopy(swarm)
    best_scores_vector = []

    best_iteration = None
    current_best_score = float('inf')
    
    # MAIN LOOP
    for it in range(mandatory.max_iterations):
        print("iter. no. ", it)

        # update parameters
        alpha = mandatory.alpha_initial * np.exp(-optional.alpha_decay * it)
        # Attractiveness varies with distance r via exp[−γr]
        gamma = mandatory.gamma # light absorption

        # check all fireflies in pairs
        for firefly_i in swarm.elements:
            for firefly_j in swarm.elements:
                if firefly_j.brightness > firefly_i.brightness:
                    firefly_i.move_towards(
                        firefly_j,
                        alpha,
                        gamma,
                        mandatory.beta0
                    )
        swarm.update_global_best()

        # Sprawdzamy, czy znaleziono nowy najlepszy wynik
        if swarm.global_best_score < current_best_score:
            current_best_score = swarm.global_best_score
            best_iteration = it  # Zapisujemy numer pierwszej iteracji z najlepszym wynikiem

        print('global best score: ', swarm.global_best_score)
        print('global best iteration: ', best_iteration)  # Wyświetlamy pierwszą iterację z najlepszym wynikiem
        
        best_scores_vector.append(swarm.global_best_score)
        
    # pylint: disable=R0801
    final_swarm = deepcopy(swarm)
    return [
        initial_swarm,
        final_swarm,
        model.chosen_states,
        model.initial_state,
        mandatory.max_iterations,
        best_scores_vector
    ]

def main():
    """
    A test instance - will not be run when algorithm used by GUI
    """
    firefly_alg(MandatorySettingsFA())


if __name__ == '__main__':
    main()
