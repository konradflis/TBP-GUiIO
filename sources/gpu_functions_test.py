import numpy as np
from numba import cuda
from time import time

from common_elements import ModelProperties
from constant import DATA_PATH_ORBIT_L2_7
from gpu_functions import rk4_step_cpu, propagate_gpu, select_parents, crossover, mutate

# Set the parameters
pop_size = 100
chrom_size = 6
num_generations = 5
model = ModelProperties(filepath=DATA_PATH_ORBIT_L2_7, number_of_measurements=35)
mu = 0.01215058560962404
dt = 0.002
num_steps = 800
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

    # Display the value of the best fitness in the generation
    best_fitness = np.min(fitnesses)
    print(f"Best fitness: {best_fitness}")
    # print(fitnesses)

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
