"""
data_load.py module is responsible for data preprocessing and conversions.
"""
import pandas as pd
import numpy as np


def transfer_raw_data_to_trajectory(
        file_path,
        number_of_measurements):
    """
    Transfers the data downloaded from NASA database into formatted data vectors.
    :param file_path: text file's path containing original orbit's measurements (NASA database)
    :param number_of_measurements: number of time samples when the position
    is calculated and compared
    :return: set of information (time vector, selected points from the input trajectory file)
    necessary for further calculations
    """
    # pylint: disable=R0914
    df = pd.read_csv(file_path)
    time_vect = df['Time (TU)']
    states_pos = df.loc[:, 'X (LU)': 'Z (LU)']
    states_vel = df.loc[:, 'VX (LU/TU)': 'VZ (LU/TU)']
    all_states = pd.concat([states_pos, states_vel], axis=1)
    df_new_units = pd.concat([time_vect, states_pos, states_vel], axis=1)
    initial_state = np.array(all_states.loc[0])
    step = len(all_states) / number_of_measurements
    idx = [int(i * step) for i in range(number_of_measurements)]
    chosen_states = df_new_units.loc[idx].reset_index(drop=True)
    return time_vect.iloc[-1], all_states, initial_state, chosen_states


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
