from numba import cuda, float32, config

import numpy as np
import random
from copy import deepcopy
from typing import List

config.CUDA_ENABLE_PYNVJITLINK = 1


def verify_device_access():

    """
    Check is your GPU is accessible.
    """

    if cuda.is_available():

        device = cuda.get_current_device()
        print(f"GPU name: {device.name}")
        print(f"Number of microprocessors: {device.MULTIPROCESSOR_COUNT}")
        print(f"Compute capability: {device.COMPUTE_CAPABILITY_MAJOR}.{device.COMPUTE_CAPABILITY_MINOR}")

    else:
        print("No devices found.")


def diff_eq_cpu(state, mu):

    """
    Define differential equations for CPU computing.
    :param state: current state
    :param mu: center of gravity
    :return: the next state
    """

    x1, y1, z1, x2, y2, z2 = state
    r_se = ((x1 + mu) ** 2 + y1 ** 2 + z1 ** 2) ** 1.5
    r_sm = ((x1 + mu - 1) ** 2 + y1 ** 2 + z1 ** 2) ** 1.5

    out_x2 = 2 * y2 + x1 - (1 - mu) * (x1 + mu) / r_se - mu * (x1 + mu - 1) / r_sm
    out_y2 = -2 * x2 + y1 - (1 - mu) * y1 / r_se - mu * y1 / r_sm
    out_z2 = - (1 - mu) * z1 / r_se - mu * z1 / r_sm

    return np.array([x2, y2, z2, out_x2, out_y2, out_z2], dtype=np.float32)


def rk4_step_cpu(state, dt, mu):

    """
    Solver for CPU computing -- 4th order Range-Kutta method.
    :param state: current state
    :param dt: solver step
    :param mu: center of gravity
    :return: next state
    """

    k1 = diff_eq_cpu(state, mu)
    k2 = diff_eq_cpu(state + 0.5 * dt * k1, mu)
    k3 = diff_eq_cpu(state + 0.5 * dt * k2, mu)
    k4 = diff_eq_cpu(state + dt * k3, mu)

    next_state = state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return next_state


@cuda.jit(device=True)
def diff_eq(statex, mu, output):

    """
    Define differential equations for GPU computing.
    :param statex: current state
    :param mu: center of gravity
    :param output: next state
    """

    x1, y1, z1, x2, y2, z2 = statex
    r_se = ((x1 + mu) ** 2 + y1 ** 2 + z1 ** 2) ** 1.5  # satellite -- earth distance
    r_sm = ((x1 + mu - 1) ** 2 + y1 ** 2 + z1 ** 2) ** 1.5  # satellite -- moon distance

    output[0] = x2
    output[1] = y2
    output[2] = z2
    output[3] = 2 * y2 + x1 - (1 - mu) * (x1 + mu) / r_se - mu * (x1 + mu - 1) / r_sm
    output[4] = -2 * x2 + y1 - (1 - mu) * y1 / r_se - mu * y1 / r_sm
    output[5] = - (1 - mu) * z1 / r_se - mu * z1 / r_sm


@cuda.jit(device=True)
def rk4_step(statex, dt, mu):

    """
    Solver for GPU computing -- 4th order Range-Kutta method.
    :param statex: current state
    :param dt: solver step
    :param mu: center of gravity
    """

    # Allocate memory for slopes
    k1 = cuda.local.array(shape=6, dtype=float32)
    k2 = cuda.local.array(shape=6, dtype=float32)
    k3 = cuda.local.array(shape=6, dtype=float32)
    k4 = cuda.local.array(shape=6, dtype=float32)
    temp_state = cuda.local.array(shape=6, dtype=float32)

    # Compute k1 = f(state)
    diff_eq(statex, mu, k1)

    # Compute k2 = f(state + 0.5 * dt * k1)
    for i in range(6):
        temp_state[i] = statex[i] + 0.5 * dt * k1[i]
    diff_eq(temp_state, mu, k2)

    # Compute k3 = f(state + 0.5 * dt * k2)
    for i in range(6):
        temp_state[i] = statex[i] + 0.5 * dt * k2[i]
    diff_eq(temp_state, mu, k3)

    # Compute k4 = f(state + dt * k3)
    for i in range(6):
        temp_state[i] = statex[i] + dt * k3[i]
    diff_eq(temp_state, mu, k4)

    # Compute next state: state = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    for i in range(6):
        statex[i] += (dt / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])


@cuda.jit
def propagate_gpu(states, steps, pop_size, original_trajectory, fitnesses):

    """
    For loop on GPU.
    :param states: trajectory
    :param steps: number of steps
    :param pop_size: rozmiar populacji
    :param original_trajectory: oryginalna orbita
    :param fitnesses: wartości funkcji celu
    :return:
    """

    tx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if tx >= pop_size:
        return

    mu = 0.01215058560962404
    dt = 0.001

    statex = cuda.local.array(6, dtype=float32)

    for i in range(6):
        statex[i] = states[tx, i]  # copy the current state

    for t in range(steps):
        rk4_step(statex, dt, mu)

        fitness = 0
        for i in range(3):
            target_index = t * 6 + i
            fitness += (statex[i] - original_trajectory[target_index]) ** 2

        fitnesses[tx] = fitness ** 0.5


def select_parents(individuals, fitnesses, option: bool = True, tournament_size: int = None):

    """
    Modified version of selecting parents, adjusted for GPU computing.
    :param individuals: set of individuals
    :param fitnesses: corresponding values of the objective function
    :param option: determines which type of selection is used (default True for tournament selection)
    :param tournament_size: number of individuals taking part in the tournament (doesn't need to be specified)
    :return: two parents
    """
    """
    Two select options are available:
    - option 1: after implementation add describtion
    - option 2: after implementation add describtion
    """

    # Choose the right selection method
    if option:
        return tournament_selection(individuals, fitnesses, tournament_size)
    else:
        return roulette_selection(individuals, fitnesses)


def tournament_selection(individuals, fitnesses, tournament_size: int):

    """
    Select parents via tournament method.
    :param individuals: set of individuals
    :param fitnesses: corresponding values of the objective function
    :param tournament_size: number of individuals taking part in the tournament (doesn't need to be specified)
    :return: two parents
    """

    # Copy the original arrays and converting them into lists
    copy_list_of_individuals = deepcopy(list(individuals))
    if type(copy_list_of_individuals[0]) is not List:
        for i in range(len(copy_list_of_individuals)):
            copy_list_of_individuals[i] = list(copy_list_of_individuals[i])
    copy_fitnesses = deepcopy(list(fitnesses))

    # Set the tournament size
    if tournament_size is None:
        tournament_size = len(individuals) // 2

    # Choose a random sample of scores
    tournament_1 = random.sample(copy_fitnesses, tournament_size)

    # Select the lowest score
    parent_1_score = min(tournament_1)

    # Find the index of the lowest score and pick the corresponding parent
    parent_1_inx = copy_fitnesses.index(parent_1_score)
    parent_1 = copy_list_of_individuals[parent_1_inx]

    # Remove the selected parent and its score from the second tournament
    copy_list_of_individuals.remove(parent_1)
    copy_fitnesses.remove(parent_1_score)

    # Pick the second parent in the same manner
    tournament_2 = random.sample(copy_fitnesses, tournament_size)
    parent_2_score = min(tournament_2)
    parent_2_inx = copy_fitnesses.index(parent_2_score)
    parent_2 = copy_list_of_individuals[parent_2_inx]

    return list(parent_1), list(parent_2)


def roulette_selection(individuals, fitnesses):

    """
    Select parents via roulette method.
    :param individuals: set of individuals
    :param fitnesses: corresponding values of the objective function
    :return: two parents
    """

    # Copy the original arrays and converting them into lists
    copy_list_of_individuals = deepcopy(list(individuals))
    if type(copy_list_of_individuals[0]) is not List:
        for i in range(len(copy_list_of_individuals)):
            copy_list_of_individuals[i] = list(copy_list_of_individuals[i])

    copy_fitnesses = deepcopy(list(fitnesses))

    # If all fitnesses are equal to 0, then pick two random parents
    total_fitness = np.sum(copy_fitnesses)
    if total_fitness == 0:
        return tuple(random.sample(copy_list_of_individuals, 2))

    # Calculate the probability of selection for each individual
    selection_probs_1 = [copy_fitnesses[i] / total_fitness for i in range(len(copy_fitnesses))]

    # Select the individual index
    selected_index_1 = random.choices(range(len(copy_list_of_individuals)), weights=selection_probs_1, k=1)[0]

    # Set the individual as the first parent, remove the individual and its fitness from the lists
    parent_1 = copy_list_of_individuals.pop(selected_index_1)
    copy_fitnesses.pop(selected_index_1)

    # The same steps for the second parent
    selection_probs_2 = [copy_fitnesses[i] / total_fitness for i in range(len(copy_fitnesses))]
    selected_index_2 = random.choices(range(len(copy_list_of_individuals)), weights=selection_probs_2, k=1)[0]
    parent_2 = copy_list_of_individuals.pop(selected_index_2)
    copy_fitnesses.pop(selected_index_2)

    return parent_1, parent_2


def crossover(parent1, parent2, crossover_rate=0.7):

    """
    Modified version of crossover, adjusted to GPU computing.
    :param parent1: individual -- parent
    :param parent2: individual -- parent
    :param crossover_rate: rate of crossover happening (doesn't need to be specified)
    :return: two offspring
    """

    # Set the probability of crossover
    probability = random.random()

    # Choose the right crossover method
    if probability < crossover_rate / 2:
        return crossover_1(parent1, parent2)
    elif crossover_rate / 2 <= probability < crossover_rate:
        return crossover_2(parent1, parent2)
    else:
        return crossover_copy(parent1, parent2)


def crossover_1(parent1, parent2):

    """
    Random crossover: Each gene has a chance to come from either parent. Avoids copying all genes from one parent.
    :param parent1: individual -- parent
    :param parent2: individual -- parent
    :return: two offspring
    """

    # Decide which gene comes from which parent
    offspring1_poz_idx = random.choices([1, 2], k=6)
    while offspring1_poz_idx == [1] * 6 or offspring1_poz_idx == [2] * 6:
        offspring1_poz_idx = random.choices([1, 2], k=6)
    offspring2_poz_idx = [2 if x == 1 else 1 for x in offspring1_poz_idx]

    # Create offspring
    offspring1 = deepcopy([parent1[i] if par == 1 else parent2[i] for i, par in enumerate(offspring1_poz_idx)])
    offspring2 = deepcopy([parent1[i] if par == 1 else parent2[i] for i, par in enumerate(offspring2_poz_idx)])
    return offspring1, offspring2


def crossover_2(parent1, parent2):

    """
    Single-point crossover: Genes are exchanged at a random point.
    :param parent1: individual -- parent
    :param parent2: individual -- parent
    :return: two offspring
    """

    # Pick the crossover point
    point = random.randint(1, 5)

    # Create offspring
    offspring1 = deepcopy(parent1[:point] + parent2[point:])
    offspring2 = deepcopy(parent2[:point] + parent1[point:])
    return offspring1, offspring2


def crossover_copy(parent1, parent2):

    """
    No crossover: Offspring are direct copies of the parents.
    :param parent1: individual -- parent
    :param parent2: individual -- parent
    :return: two offspring
    """

    # Create offspring (copy parents)
    offspring1 = deepcopy(parent1)
    offspring2 = deepcopy(parent2)
    return offspring1, offspring2


def mutate(individual, mutation_rate=0.01):

    """
    Modified version of mutation, adjusted to GPU computing.
    :param individual: individual
    :param mutation_rate: rate of mutation (doesn't need to be specified)
    """

    # If mutation is going to happen, pick a method
    option = True
    if random.random() < mutation_rate:
        option = random.choice([True, False])

    if option:
        mutate_1(individual)
    else:
        mutate_2(individual)


def mutate_1(individual):

    """
    One of the two mutate options. Method that mutates a single gene with constant shift.
    :param individual: individual
    """

    mutate_index = random.randint(0, 5)
    shift = 0.001
    individual[mutate_index] += shift


def mutate_2(individual):

    """
    One of the two mutate options. Method that mutates random numbers genes with random shift.
    """

    indexes = random.sample([0, 1, 2, 3, 4, 5], k=random.randint(2, 6))
    for i in indexes:
        shift = random.uniform(-0.001, 0.001)
        individual[i] += shift
