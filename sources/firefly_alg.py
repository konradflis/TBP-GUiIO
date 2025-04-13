"""
Firefly Algorithm implementation for orbital trajectory optimization.

Based on Xin-She Yang's Firefly Algorithm (2009) & adaptations for 3 body problem
"""
from copy import deepcopy
import numpy as np
from numba import cuda, float64, config
import math

config.CUDA_ENABLE_PYNVJITLINK = 1

from sources.common_elements import PropagatedElement, ModelProperties, Swarm
from sources.data_structures import MandatorySettingsFA, OptionalSettingsFA


# -------------------- GPU Device Functions --------------------
@cuda.jit(device=True)
def diff_eq_gpu(statex, mu, out):
    """
    Differential equations for the CR3BP (GPU) in double precision.
    statex: [x, y, z, vx, vy, vz]
    mu: CR3BP mu parameter
    out: derivative [vx, vy, vz, ax, ay, az]
    """
    x1, y1, z1, x2, y2, z2 = statex

    r_se = ((x1 + mu) ** 2 + y1 ** 2 + z1 ** 2) ** 1.5
    r_sm = ((x1 + mu - 1.0) ** 2 + y1 ** 2 + z1 ** 2) ** 1.5

    out[0] = x2
    out[1] = y2
    out[2] = z2
    out[3] = 2.0 * y2 + x1 - (1.0 - mu) * (x1 + mu) / r_se - mu * (x1 + mu - 1.0) / r_sm
    out[4] = -2.0 * x2 + y1 - (1.0 - mu) * y1 / r_se - mu * y1 / r_sm
    out[5] = -(1.0 - mu) * z1 / r_se - mu * z1 / r_sm


@cuda.jit(device=True)
def rk4_step_gpu(statex, dt, mu):
    """
    4th order Runge-Kutta solver step (GPU) in double precision.
    statex: current state [x, y, z, vx, vy, vz]
    dt: time step
    mu: CR3BP mu parameter
    """
    k1 = cuda.local.array(6, float64)
    k2 = cuda.local.array(6, float64)
    k3 = cuda.local.array(6, float64)
    k4 = cuda.local.array(6, float64)
    temp_state = cuda.local.array(6, float64)

    # Calculate k1
    diff_eq_gpu(statex, mu, k1)

    # Calculate k2
    for i in range(6):
        temp_state[i] = statex[i] + 0.5 * dt * k1[i]
    diff_eq_gpu(temp_state, mu, k2)

    # Calculate k3
    for i in range(6):
        temp_state[i] = statex[i] + 0.5 * dt * k2[i]
    diff_eq_gpu(temp_state, mu, k3)

    # Calculate k4
    for i in range(6):
        temp_state[i] = statex[i] + dt * k3[i]
    diff_eq_gpu(temp_state, mu, k4)

    # Combine the slopes to update the state
    for i in range(6):
        statex[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])


@cuda.jit
def cost_kernel_gpu(states_d, chosen_states_d, steps, mu, cost_d):
    """
    Processes each firefly in parallel. For the firefly at index tx:
      - Copies its state from states_d[tx, :]
      - Propagates this state using RK4 with M substeps for each measurement interval,
      - After the propagation, compares the current state with the corresponding
        reference state (chosen_states_d) and accumulates the squared differences (for the positions),
      - After completing all measurement steps, writes the square root of the accumulated error to cost_d[tx].
    """
    tx = cuda.grid(1)          # Global thread index (1D)
    pop_size = states_d.shape[0]

    if tx >= pop_size:
        return  # Out of range

    # Local copy of the state for this firefly
    s = cuda.local.array(6, float64)
    for i in range(6):
        s[i] = states_d[tx, i]

    cost_val = 0.0

    # Propagation parameters:
    # Set total propagation time (can later be replaced with a parameter from the model)
    total_time = 100.0  
    dt_measure = total_time / steps   # Time for each measurement interval
    M = 10                            # Number of RK4 substeps per measurement interval
    dt_sub = dt_measure / M           # Time step for each substep

    # For each measurement step:
    for t in range(steps):
        # Propagate the state through the measurement interval using M substeps
        for _ in range(M):
            rk4_step_gpu(s, dt_sub, mu)
            
        # After propagation, measure the error between the current position and the reference state
        dx = s[0] - chosen_states_d[t, 0]
        dy = s[1] - chosen_states_d[t, 1]
        dz = s[2] - chosen_states_d[t, 2]
        cost_val += (dx * dx + dy * dy + dz * dz)

    # Write the final objective function value (square root of the accumulated error)
    cost_d[tx] = math.sqrt(cost_val)


# -------------------- Swarm / Firefly (GPU-based) --------------------
class SwarmFirefly(Swarm):
    """
    Creates a swarm of fireflies (agents).
    Defines the basic characteristics of the swarm.
    """
    def generate_initial_population(self,
                                    opt_if_two_stage_pso=0,
                                    opt_best_velocity=None):
        initial_random = super().generate_initial_population()
        for idx in range(self.population_size):
            firefly = Firefly(initial_random[idx], self, self.model)
            self.elements.append(firefly)


class Firefly(PropagatedElement):
    """
    Defines a firefly and all its properties.
    
    NOTE: We do not calculate the cost individually on the GPU for each firefly,
          because we compute it in batch via cost_kernel_gpu for the entire swarm.
    """
    def __init__(self, state, swarm, model):
        super().__init__(state, model)
        self.swarm = swarm
        self.score = float('inf')
        self.brightness = 0.0

    def move_towards(self, other, alpha, gamma, beta0, attractiveness_function, movement_type):
        """
        Firefly movement with auto-matched randomization ranges,
        using the same limits as in the swarm's initialization.
        (Does NOT recalculate cost individually; it will be computed in batch.)
        """
        # Calculate the Euclidean distance between this firefly and the other
        r = np.linalg.norm(self.state - other.state)

        # Compute attractiveness based on the specified function
        if attractiveness_function == 0:  # Exponential
            beta = beta0 * np.exp(-gamma * r ** 2)
        elif attractiveness_function == 1:  # Inverse square
            beta = beta0 / (1 + gamma * r ** 2)
        elif attractiveness_function == 2:  # Sigmoid-like
            beta = beta0 / (1 + np.exp(gamma * (r - 1)))
        else:
            raise ValueError("Unknown attractiveness_function")

        # Generate a random vector within the bounds of the swarm (using the first firefly as reference)
        temp_swarm = Swarm(1, 1, self.model)
        temp_pop = temp_swarm.generate_initial_population()
        random_vector = temp_pop[0] - self.model.initial_state

        # Compute the new state based on the chosen movement type
        if movement_type == 0:
            new_state = self.state + beta * (other.state - self.state) + alpha * random_vector
        elif movement_type == 1:
            exp_attract = beta * np.exp(-gamma * r ** 2)
            new_state = self.state + exp_attract * (other.state - self.state) + alpha * (random_vector - 0.5)
        elif movement_type == 2:
            gaussian_noise = np.random.normal(0, 1, size=self.state.shape)
            new_state = self.state + beta * (other.state - self.state) + alpha * gaussian_noise
        else:
            raise ValueError("Unknown movement_type")

        # Clamp the new state to the specified bounds (using a freshly generated firefly as reference)
        clipped_state = temp_swarm.generate_initial_population()[0]
        self.state = new_state + (clipped_state - temp_pop[0])
        # Cost is not recalculated here; it is computed in batch later.


# -------------------- Batch Cost Calculation --------------------
def batch_calculate_costs(swarm, model):
    """
    Compute the cost for ALL fireflies in the swarm in a single kernel call.
    This uses threads_per_block = 1024 and enough blocks to cover the entire population.
    Afterwards, each firefly's .score and .brightness are updated on the host.
    """
    pop_size = len(swarm.elements)
    if pop_size == 0:
        return  # No fireflies

    # Build a 2D states array on the host: shape (pop_size, 6)
    states_h = np.array([f.state for f in swarm.elements], dtype=np.float64)
    chosen_h = model.chosen_states.astype(np.float64)
    steps = chosen_h.shape[0]

    # Copy data to the device
    states_d = cuda.to_device(states_h)
    chosen_d = cuda.to_device(chosen_h)
    cost_d = cuda.device_array(pop_size, dtype=np.float64)

    # Launch the kernel
    threads_per_block = 1024
    if pop_size < threads_per_block:
        # Use the smaller number to avoid mostly empty blocks
        threads_per_block = pop_size
    blocks = (pop_size + threads_per_block - 1) // threads_per_block
    cost_kernel_gpu[blocks, threads_per_block](states_d, chosen_d, steps,
                                               np.float64(0.01215058560962404),
                                               cost_d)

    # Copy the resulting cost array back to the host
    host_cost = cost_d.copy_to_host()

    # Update each firefly's score and brightness
    for idx, firefly in enumerate(swarm.elements):
        firefly.score = float(host_cost[idx])
        firefly.brightness = 1.0 / (firefly.score + 1e-10)


# -------------------- Main Firefly Function --------------------
def firefly_alg(mandatory, optional=None):
    """
    Core implementation of the Firefly Algorithm for orbital trajectory optimization.
    
    Parameters:
      mandatory: structure containing mandatory parameters (e.g., max_iterations, population_size, etc.)
      optional: structure containing optional parameters (e.g., alpha_decay, attractiveness_function, etc.)
    
    Returns:
      A list containing:
        - initial_swarm: the initial population of fireflies
        - final_swarm: the final swarm after optimization
        - chosen_states: the reference trajectory (from the model)
        - initial_state: the initial state from the model
        - max_iterations: the maximum number of iterations
        - best_scores_vector: a list of global best cost values per iteration
    """
    if optional is None:
        optional = OptionalSettingsFA()

    # Initialize the model for orbital trajectory optimization
    model = ModelProperties(optional.orbit_filepath, mandatory.number_of_measurements)

    # Initialize the swarm
    swarm = SwarmFirefly(mandatory.population_size, mandatory.max_iterations, model)
    swarm.generate_initial_population()

    # Calculate initial cost for the swarm in batch mode
    batch_calculate_costs(swarm, model)
    swarm.update_global_best()
    initial_swarm = deepcopy(swarm)

    best_scores_vector = []
    best_iteration = None
    current_best_score = float('inf')

    # Main optimization loop
    for it in range(mandatory.max_iterations):
        print("iter. no. ", it)
        alpha = mandatory.alpha_initial * np.exp(-optional.alpha_decay * it)
        gamma = mandatory.gamma

        # Compare fireflies and move them based on brightness
        if optional.compare_type == 0:  # 'all-all'
            for f_i in swarm.elements:
                for f_j in swarm.elements:
                    if f_j.brightness > f_i.brightness:
                        f_i.move_towards(f_j, alpha, gamma,
                                         mandatory.beta0,
                                         optional.attractiveness_function,
                                         optional.movement_type)
        elif optional.compare_type == 1:  # 'all-all-no-duplicates'
            for idx, f_i in enumerate(swarm.elements):
                for f_j in swarm.elements[idx + 1:]:
                    if f_j.brightness > f_i.brightness:
                        f_i.move_towards(f_j, alpha, gamma,
                                         mandatory.beta0,
                                         optional.attractiveness_function,
                                         optional.movement_type)
                    else:
                        f_j.move_towards(f_i, alpha, gamma,
                                         mandatory.beta0,
                                         optional.attractiveness_function,
                                         optional.movement_type)
        elif optional.compare_type == 2:  # 'by-pairs'
            for idx in range(len(swarm.elements) - 1):
                if swarm.elements[idx].brightness > swarm.elements[idx + 1].brightness:
                    swarm.elements[idx + 1].move_towards(
                        swarm.elements[idx], alpha, gamma,
                        mandatory.beta0,
                        optional.attractiveness_function,
                        optional.movement_type
                    )
                else:
                    swarm.elements[idx].move_towards(
                        swarm.elements[idx + 1], alpha, gamma,
                        mandatory.beta0,
                        optional.attractiveness_function,
                        optional.movement_type
                    )
        else:
            raise ValueError("Unknown compare_type")

        # Recalculate costs for all fireflies (batch mode) after movements
        batch_calculate_costs(swarm, model)
        swarm.update_global_best()

        # Track the global best cost
        if swarm.global_best_score < current_best_score:
            current_best_score = swarm.global_best_score
            best_iteration = it

        print('global best score: ', swarm.global_best_score)
        print('global best iteration: ', best_iteration)
        best_scores_vector.append(swarm.global_best_score)

    # Final swarm after optimization
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
    firefly_alg(MandatorySettingsFA())


if __name__ == '__main__':
    main()
