"""
data_load.py module is responsible for data preprocessing and conversions.
"""
import pandas as pd
import numpy as np


def transfer_raw_data_to_trajectory(
        file_path,
        population_size,
        number_of_measurements,
        opt_if_two_stage_pso=0,
        opt_best_velocity=None):
    """
    Transfers the data downloaded from NASA database into formatted data vectors.
    :param file_path: text file's path containing original orbit's measurements (NASA database)
    :param population_size: number of particles/sources (swarm's elements)
    :param number_of_measurements: number of time samples when the position
    is calculated and compared
    :param opt_if_two_stage_pso: optional: defines the way points are generated in the search space
    :param opt_best_velocity: optional: defines if best found velocity so far should impact
    the new set of initial particles
    :return: set of information (time vector, selected points from the input trajectory file)
    necessary for further calculations
    """
    # pylint: disable=R0914
    lu_to_km_coeff = 389703
    tu_to_s_coeff = 382981
    pos_limits = 250 / lu_to_km_coeff
    vel_limits = 0.1 / (lu_to_km_coeff / tu_to_s_coeff)
    df = pd.read_csv(file_path)
    time_vect = df['Time (TU)']
    states_pos = df.loc[:, 'X (LU)': 'Z (LU)']
    states_vel = df.loc[:, 'VX (LU/TU)': 'VZ (LU/TU)']
    states = pd.concat([states_pos, states_vel], axis=1)
    df_new_units = pd.concat([time_vect, states_pos, states_vel], axis=1)
    initial_state = np.array(states.loc[0])
    initial_random = []

    if not opt_if_two_stage_pso:
        for idx in range(population_size):
            position = [np.random.uniform(-pos_limits, pos_limits) for _ in range(3)]
            velocity = [np.random.uniform(-vel_limits, vel_limits) for _ in range(3)]
            random_vect = position + velocity
            initial_random.append(initial_state + random_vect)
    else:
        for idx in range(population_size):
            position = [np.random.uniform(-pos_limits, pos_limits) for _ in range(3)]
            velocity = [0, 0, 0]
            random_vect = position + velocity
            initial_random.append(initial_state + random_vect)

        for elem in initial_random:
            percent = 0.001

            elem[3:] = [opt_best_velocity[0] + np.random.uniform(-1 * percent, percent),
                        opt_best_velocity[1] + np.random.uniform(-1 * percent, percent),
                        opt_best_velocity[2] + np.random.uniform(-1 * percent, percent)]

    step = len(states) / number_of_measurements
    idx = [int(i * step) for i in range(number_of_measurements)]
    chosen_states = df_new_units.loc[idx].reset_index(drop=True)
    return time_vect.iloc[-1], states, initial_state, initial_random, chosen_states


def convert_to_metric_units(vect):
    """
    Converts the original data units to metric system.
    :param vect: data in NASA database units (TU, LU, LU/TU etc.)
    :return: data in metric units
    """
    lu_to_km_coeff = 389703
    tu_to_s_coeff = 382981
    vect[:3] = vect[:3] * lu_to_km_coeff
    if len(vect) > 3:
        vect[3:] = vect[3:] * (lu_to_km_coeff / tu_to_s_coeff)
    return vect
