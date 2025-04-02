"""
plot_functions.py module provides with functions that are used to plot trajectories.
"""
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from sources.data_load import convert_to_metric_units


def dim3_scatter_plot(
        series1,
        series2,
        initial_state,
        global_best_state):
    """
    Creates the scatter plots of positions and velocities (comparing the initial and final swarms)
    :param series1: initial positions
    :param series2: final positions
    :param initial_state: initial original state
    :param global_best_state: the state with the lowest objective function value
    :return: scatter plots fig1, fig2
    """
    fig1 = plt.figure()
    ax = fig1.add_subplot(projection='3d')

    xs1 = [el[0] for el in series1]
    ys1 = [el[1] for el in series1]
    zs1 = [el[2] for el in series1]
    ax.scatter(xs1, ys1, zs1, color='g', label='położenia początkowe')

    xs2 = [el[0] for el in series2]
    ys2 = [el[1] for el in series2]
    zs2 = [el[2] for el in series2]

    ax.scatter(xs2, ys2, zs2, color='b', label='położenia końcowe')

    ax.scatter(
        initial_state[0],
        initial_state[1],
        initial_state[2],
        color='r',
        marker='v')
    ax.scatter(
        global_best_state[0],
        global_best_state[1],
        global_best_state[2],
        color='k',
        marker='v',
        s=20)

    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_zlabel('Z [km]')
    plt.legend(loc='upper left')

    ax.set_xlim(initial_state[0] - 250, initial_state[0] + 250)
    ax.set_ylim(initial_state[1] - 250, initial_state[1] + 250)
    ax.set_zlim(initial_state[2] - 250, initial_state[2] + 250)

    fig2 = plt.figure()
    ax = fig2.add_subplot(projection='3d')

    xs1 = [el[3] for el in series1]
    ys1 = [el[4] for el in series1]
    zs1 = [el[5] for el in series1]
    ax.scatter(xs1, ys1, zs1, color='g', label='prędkości początkowe')

    xs2 = [el[3] for el in series2]
    ys2 = [el[4] for el in series2]
    zs2 = [el[5] for el in series2]

    ax.scatter(xs2, ys2, zs2, color='b', label='prędkości końcowe')

    ax.scatter(
        initial_state[3],
        initial_state[4],
        initial_state[5],
        color='r',
        marker='v')

    ax.set_xlabel('X [km/s]')
    ax.set_ylabel('Y [km/s]')
    ax.set_zlabel('Z [km/s]')

    ax.set_xlim(initial_state[3] - 0.1, initial_state[3] + 0.1)
    ax.set_ylim(initial_state[4] - 0.1, initial_state[4] + 0.1)
    ax.set_zlim(initial_state[5] - 0.1, initial_state[5] + 0.1)

    plt.legend(loc='upper left')

    return fig1, fig2


def plot_global_best_scores(data_vector, iterations):
    """
    Creates the plot of best score in a given iteration.
    :param data_vector: best score in each iteration
    :param iterations: number of iterations
    :return: created plot
    """
    fig = plt.figure()
    plt.plot(list(range(iterations)), data_vector)
    plt.yscale('log')
    plt.xlabel('Numer iteracji')
    plt.ylabel('Wartość błędu/funkcji celu [LU]')
    return fig


def plot_propagated_trajectories(
        original_trajectory,
        propagated_trajectory,
        original_initial,
        propagated_initial):
    """
    Creates a plot of both original and found trajectories.
    :param original_trajectory: vector of original orbit states at given time
    :param propagated_trajectory: vector of found orbit states at given time
    :param original_initial: the first state of the original orbit
    :param propagated_initial: the first state of the found orbit
    :return: created plot
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    x_original = list(original_trajectory[0])
    y_original = list(original_trajectory[1])
    z_original = list(original_trajectory[2])
    ax.scatter(x_original, y_original, z_original,
               color='g', label='oryginalna orbita')

    x_propagated = list(propagated_trajectory[0])
    y_propagated = list(propagated_trajectory[1])
    z_propagated = list(propagated_trajectory[2])

    ax.scatter(x_propagated, y_propagated, z_propagated,
               color='b', label='otrzymana orbita')

    ax.scatter(
        original_initial[0],
        original_initial[1],
        original_initial[2],
        color='g',
        marker='v')
    ax.scatter(
        propagated_initial[0],
        propagated_initial[1],
        propagated_initial[2],
        color='b',
        marker='v')

    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_zlabel('Z [km]')
    ax.legend(loc='upper left')

    return fig


def present_results(initial, propagated):
    """
    Presents results in a text form, comparing basic characteristics with the expected ones.
    :param initial: state vector - first point of the original orbit
    :param propagated: state vector - first point of the found orbit
    :return: set of information about the original and propagated trajectories
    """
    print("========RESULTS==========")
    print("x: original:", initial[0], " vs propagated: ", propagated[0])
    print("y: original:", initial[1], " vs propagated: ", propagated[1])
    print("z: original:", initial[2], " vs propagated: ", propagated[2])
    print("vx: original:", initial[3], " vs propagated: ", propagated[3])
    print("vy: original:", initial[4], " vs propagated: ", propagated[4])
    print("vz: original:", initial[5], " vs propagated: ", propagated[5])
    print("=========================")
    initial_vect = convert_to_metric_units(deepcopy(np.array(initial)))
    propagated_vect = convert_to_metric_units(deepcopy(np.array(propagated)))
    diff_vect = abs(initial_vect - propagated_vect)
    dist_pos = np.linalg.norm(diff_vect[:3])
    dist_vel = np.linalg.norm(diff_vect[3:])
    dist_vect = [dist_pos, 0, 0, dist_vel, 0, 0]
    result_matrix = np.array(
        [initial_vect, propagated_vect, diff_vect, dist_vect])
    return result_matrix
