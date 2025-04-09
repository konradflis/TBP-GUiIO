import copy

import numpy as np
from numba import cuda
from time import time
import matplotlib.pyplot as plt

from common_elements import ModelProperties
from constant import DATA_PATH_ORBIT_L2_7
from gpu_functions import rk4_step_cpu, propagate_gpu, select_parents, crossover, mutate
from data_load import convert_to_metric_units
from plot_functions import plot_propagated_trajectories, dim3_scatter_plot


# Set the parameters
pop_size = 100
chrom_size = 6
num_generations = 10
model = ModelProperties(filepath=DATA_PATH_ORBIT_L2_7, number_of_measurements=35)
mu = 0.01215058560962404
dt = 0.001
num_steps = 1600
lu_to_km_coeff = 389703
tu_to_s_coeff = 382981
pos_limits = 250 / lu_to_km_coeff
vel_limits = 0.1 / (lu_to_km_coeff / tu_to_s_coeff)
crossover_rate = 0.6
mutation_rate = 0.05

# Set the initial trajectory
original_trajectory = np.zeros((num_steps, 6))
state = model.initial_state.copy()
for t in range(num_steps):
    original_trajectory[t] = state
    state = rk4_step_cpu(state, dt, mu)

# Set the chromosomes in the initial population
chromosomes = []

for _ in range(pop_size):
    position = [np.random.uniform(-pos_limits, pos_limits) for _ in range(3)]
    velocity = [np.random.uniform(-vel_limits, vel_limits) for _ in range(3)]
    random_vect = position + velocity
    chromosomes.append(model.initial_state + random_vect)

chromosomes = np.array(chromosomes)

# Prepare kernel
threads_per_block = 256
blocks_per_grid = (pop_size + (threads_per_block - 1)) // threads_per_block

# Initialise arrays
fitnesses = np.zeros(pop_size, dtype=np.float32)
target_positions = np.array(original_trajectory, dtype=np.float32, order='C')
target_positions = np.ravel(target_positions)

# Initialise lists for saving the best results
best_individuals = []
best_individuals_converted = []
best_fitnesses = []

# Copy the initial trajectory to GPU
d_target_positions = cuda.to_device(target_positions)

start = time()

# Run the genetic algorithm
for i in range(num_generations):
    print(f"Gen {i}/{num_generations-1}")

    d_chromosomes = cuda.to_device(chromosomes)     # Copy the set of initial states to GPU
    d_fitnesses = cuda.to_device(fitnesses)         # Copy the fitness array

    # Propagate the generation
    propagate_gpu[blocks_per_grid, threads_per_block](d_chromosomes, num_steps, pop_size, d_target_positions,
                                                      d_fitnesses)
    # Copy the results to host
    fitnesses = d_fitnesses.copy_to_host()
    chromosomes = d_chromosomes.copy_to_host()

    # Display the value of the best fitness and the corresponding individual
    best_fitness = np.min(fitnesses)
    best_inx = np.argwhere(fitnesses == best_fitness)[0][0]
    best_individual = chromosomes[best_inx]
    best_individual_converted = convert_to_metric_units(copy.deepcopy(best_individual))
    print(f"Best fitness: {best_fitness}")
    print(f"Best individual: {best_individual}")
    print(f"x:    {best_individual_converted[0]} m")
    print(f"y:    {best_individual_converted[1]} m")
    print(f"z:    {best_individual_converted[2]} m")
    print(f"v_x: {best_individual_converted[3]}  m/s")
    print(f"v_y: {best_individual_converted[4]}  m/s")
    print(f"v_z: {best_individual_converted[5]}  ms/s")

    # Save the results
    best_fitnesses.append(best_fitness)
    best_individuals.append(best_individual)
    best_individuals_converted.append(best_individual_converted)

    # Create the new generation
    new_chromosomes = []
    while len(new_chromosomes) < len(chromosomes):

        # Selection -- select new parents using the tournament method, the size of the tournament can be specified
        parent1, parent2 = select_parents(chromosomes, fitnesses)

        # Crossover -- produce offspring from the selected parents, the crossover rate can be specified
        offspring1, offspring2 = crossover(parent1, parent2, crossover_rate=crossover_rate)

        # Mutation -- potentially mutate the offspring, the mutation rate can be specified
        mutate(offspring1, mutation_rate=mutation_rate)
        mutate(offspring2, mutation_rate=mutation_rate)

        # Add the new individual to the new population
        new_chromosomes.append(offspring1)
        new_chromosomes.append(offspring2)

    # Set the next generation as the current one
    chromosomes = np.array(new_chromosomes)
    fitnesses = np.zeros(pop_size, dtype=np.float32)

end = time()
print(f"Time elapsed {end-start}")

# Pick the best solution from the run
min_fitness = min(best_fitnesses)
min_inx = best_fitnesses.index(min_fitness)
solution = best_individuals[min_inx]

# Propagate the solution (CPU)
propagated_trajectory = np.zeros((num_steps, 6))
state = solution
for t in range(num_steps):
    propagated_trajectory[t] = state
    state = rk4_step_cpu(state, dt, mu)

# Plot the propagated solution and compare it with the initial one
fig = plot_propagated_trajectories(np.transpose(original_trajectory), np.transpose(propagated_trajectory),
                                   model.initial_state.copy(), solution)
plt.show()

# Convert the original and propagated trajectory data to metric units
for i in range(original_trajectory.shape[0]):
    original_trajectory[i] = convert_to_metric_units(original_trajectory[i])
    propagated_trajectory[i] = convert_to_metric_units(propagated_trajectory[i])
    
# Plot the converted versions
fig = plot_propagated_trajectories(np.transpose(original_trajectory), np.transpose(propagated_trajectory),
                                   model.initial_state.copy(), solution)
plt.show()
