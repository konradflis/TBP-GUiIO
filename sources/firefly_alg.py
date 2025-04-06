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

    def move_towards(self, other, alpha, gamma, beta0, attractiveness_function, movement_type):
        """
        Firefly movement with auto-matched randomization ranges
        using the same limits as in Swarm's initialization
        """
        # Calculate distance and attractiveness
        r = np.linalg.norm(self.state - other.state) # cartesian distance
        if attractiveness_function == 0:
            beta = beta0 * np.exp(-gamma * r ** 2) # attractiveness function - exponential
        # decreases monotonically - inversed squared
        elif attractiveness_function == 1:
            beta = beta0 / ( 1 + gamma * r ** 2) # attractiveness function - inversed squared
        else:
            raise ValueError(
                f"Unknown attractiveness_function: '{attractiveness_function}'. "
                "Expected 'exponential' or 'quadratic_decay'."
            )

        # Generate random vector in with support of swarm (take first row to get vector)
        temp_swarm = Swarm(1, 1, self.model)
        temp_population = temp_swarm.generate_initial_population()
        random_vector = temp_population[0] - self.model.initial_state

        # All ideas Xin-She Yang mentioned are implemented below (except adaptive scaling)
        if movement_type == 0: 
            new_state = self.state + beta * (other.state - self.state) + alpha * random_vector

        elif movement_type == 1:
            exp_attractiveness = beta * np.exp(-gamma * r **2)
            new_state = self.state + exp_attractiveness * (other.state - self.state) + alpha * (random_vector - 0.5)

        elif movement_type == 2:
            gaussian_noise = np.random.normal(0, 1, size=self.state.shape)
            new_state = self.state + beta * (other.state - self.state) + alpha * gaussian_noise

        else:
            raise ValueError(
                f"Unknown movement_type: '{movement_type}'. "
                "Expected 'linear' / 'exponential' / 'gaussian' / 'scaled'"
            )


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

        # check all fireflies by all-all method
        if optional.compare_type == 0:  #'all-all':
            for firefly_i in swarm.elements:
                for firefly_j in swarm.elements:
                    if firefly_j.brightness > firefly_i.brightness:
                        firefly_i.move_towards(
                            firefly_j,
                            alpha,
                            gamma,
                            mandatory.beta0,
                            optional.attractiveness_function,
                            optional.movement_type
                        )
        elif optional.compare_type == 1: #'all-all-no-duplicates':
            for idx, firefly_i in enumerate(swarm.elements):
                for firefly_j in swarm.elements[idx + 1:]:  # Avoid duplicate comparisons
                    if firefly_j.brightness > firefly_i.brightness:
                        firefly_i.move_towards(
                            swarm.elements[idx],
                            alpha,
                            gamma,
                            mandatory.beta0,
                            optional.attractiveness_function,
                            optional.movement_type
                        )
                    else: # common case: firefly i bigger than j, reversed operation
                          # in compare to this from above
                        # in case brightnesses are equal, fireflies moves  for beta0 value
                        # in random way for any beta type
                        firefly_j.move_towards(
                            swarm.elements[idx],
                            alpha,
                            gamma,
                            mandatory.beta0,
                            optional.attractiveness_function,
                            optional.movement_type
                        )
        elif optional.compare_type == 2: # 'by-pairs':
            for idx in range(len(swarm.elements) - 1):
                    if swarm.elements[idx].brightness > swarm.elements[idx + 1].brightness:
                        swarm.elements[idx + 1].move_towards(
                            swarm.elements[idx],
                            alpha,
                            gamma,
                            mandatory.beta0,
                            optional.attractiveness_function,
                            optional.movement_type
                        )
                    else: # common case: firefly i bigger than j, reversed operation
                          # in compare to this from above
                        # in case brightnesses are equal, fireflies moves  for beta0 value
                        # in random way for any beta type
                        swarm.elements[idx].move_towards(
                            swarm.elements[idx + 1],
                            alpha,
                            gamma,
                            mandatory.beta0,
                            optional.attractiveness_function,
                            optional.movement_type
                        )
        else:
            raise ValueError(
                f"Unknown compare_type: '{optional.compare_type}'. "
                "Expected 'all-to-all', 'all-to-all-no-duplicates' or 'by-pairs'."
            )
        swarm.update_global_best()


        if swarm.global_best_score < current_best_score:
            current_best_score = swarm.global_best_score
            best_iteration = it

        print('global best score: ', swarm.global_best_score)
        print('global best iteration: ', best_iteration)

        best_scores_vector.append(swarm.global_best_score)
    # END MAIN LOOP

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
