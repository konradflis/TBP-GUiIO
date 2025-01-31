"""
This module provides with unit-tests related to PSO and ABC algorithms.
"""

import pytest
import numpy as np
from sources import pso, abc_alg
from sources.common_elements import ModelProperties, PropagatedElement, Swarm
from sources.data_structures import (MandatorySettingsPSO, OptionalSettingsPSO,
                                     MandatorySettingsABC, OptionalSettingsABC)


@pytest.mark.parametrize("algorithm, mandatory_settings, optional_settings", [
    (pso.pso, MandatorySettingsPSO(max_iterations=2), OptionalSettingsPSO()),
    (pso.pso, MandatorySettingsPSO(max_iterations=2), None),
    (abc_alg.abc_alg, MandatorySettingsABC(max_iterations=2), OptionalSettingsABC()),
    (abc_alg.abc_alg, MandatorySettingsABC(max_iterations=2), None),
])
def test_can_algorithm_be_run(algorithm, mandatory_settings, optional_settings):
    """
    Checks if an algorithm can be run with a default sets of data - both mandatory and optional.
    :param algorithm: type of algorithm (PSO/ABC)
    :param mandatory_settings: mandatory set of information
    :param optional_settings: optional set of information
    """
    result = algorithm(mandatory_settings, optional_settings)
    assert result is not None


@pytest.mark.parametrize("algorithm, mandatory_settings, optional_settings", [
    (pso.pso, MandatorySettingsPSO(max_iterations=3, number_of_measurements=25),
     OptionalSettingsPSO()),
    (abc_alg.abc_alg, MandatorySettingsABC(max_iterations=3, number_of_measurements=25),
     OptionalSettingsABC()),
])
def test_do_algorithms_return_proper_output(algorithm, mandatory_settings, optional_settings):
    """
    Checks if the algorithms outputs are of correct type/size/value.
    :param algorithm: type of algorithm (PSO/ABC)
    :param mandatory_settings: mandatory set of information
    :param optional_settings: optional set of information
    """
    result = algorithm(mandatory_settings, optional_settings)
    assert result[0] is not None #initial swarm is defined
    assert result[1] is not None #final swarm is defined
    #Number of chosen states is equal to number of measurements:
    assert len(result[2]) == mandatory_settings.number_of_measurements
    #Initial state is a 6-dim vector:
    assert len(result[3]) == 6
    #Length of top score in iteration vector is equal to number of iterations:
    assert len(result[5]) == mandatory_settings.max_iterations


@pytest.mark.parametrize("filepath, number_of_measurements", [
    ("../orbits/L2_7days.txt", 20),
    ("../orbits/ID_16.txt", 15)
])
def test_model_initialization(filepath, number_of_measurements):
    """
    Checks if the model is initialized correctly.
    :param filepath: the filepath with the orbit data.
    :param number_of_measurements: number of points in the trajectory to be compared.
    """
    model = ModelProperties(filepath, number_of_measurements)
    assert model.period > 0
    assert len(model.chosen_states) == number_of_measurements
    assert len(model.initial_state) == 6
    #Original trajectory should only have positions, not velocities:
    assert model.original_trajectory.shape[0] == 3


@pytest.mark.parametrize("filepath, number_of_measurements, state", [
    ("../orbits/L2_7days.txt", 20, 6 * [0]),
    ("../orbits/ID_16.txt", 15, 6 * [0])
])
def test_propagated_element_initialization(filepath, number_of_measurements, state):
    """
    Checks if the propagated element is initialized correctly.
    :param filepath: the filepath with the orbit data.
    :param number_of_measurements: number of points in the trajectory to be compared.
    :param state: element state (in 6-dim space).
    """
    model = ModelProperties(filepath, number_of_measurements)
    element = PropagatedElement(state, model)
    element.calculate_cost()
    assert 0 < element.score < 10000


@pytest.mark.parametrize("filepath, number_of_measurements, "
                         "opt_if_two_stage_pso, opt_best_velocity,"
                         "opt_multistart", [
    ("../orbits/L2_7days.txt", 20, 0, [0, 0, 0], 0),
    ("../orbits/L2_7days.txt", 20, 0, [0, 0, 0], 1),
    ("../orbits/ID_16.txt", 15, 1, [1, 10 , 1], 0)
])
def test_swarm_initialization(filepath, number_of_measurements,
                              opt_if_two_stage_pso, opt_best_velocity,
                              opt_multistart):
    """
    Checks if the swarm is initialized correctly.
    This test assumes that the mean velocity from 200 particles can be asserted
    within a specified range of [-0.1, +0.1].
    :param filepath: the filepath with the orbit data.
    :param number_of_measurements: number of points in the trajectory to be compared.
    :param opt_if_two_stage_pso: used by two-stage PSO to initialize the second stage with
    the best particle's velocity from the first stage.
    :param opt_best_velocity: the best velocity from the first stage.
    """
    population_size = 50
    max_iterations = 2
    model = ModelProperties(filepath, number_of_measurements)
    swarm = Swarm(population_size, max_iterations, model)
    initial_random = swarm.generate_initial_population(opt_if_two_stage_pso, opt_best_velocity)
    #Check if initial_random generated by Swarm parent class returns data of expected properties:
    assert len(initial_random) == population_size
    assert len(initial_random[0]) == 6
    mean_vel_y = np.mean([state[4] for state in initial_random])
    if opt_if_two_stage_pso:
        assert 9.9 < mean_vel_y < 10.1
    else:
        assert model.initial_state[4] - 0.1 < mean_vel_y < model.initial_state[4] + 0.1
    #For child swarm classes, see if they can use overriden parent method:
    swarm_pso = pso.SwarmPSO(max_iterations, population_size, model, 1, 1, 1)
    swarm_pso.generate_initial_population(opt_multistart=opt_multistart)
    assert len(swarm_pso.elements) == population_size
    swarm_abc = abc_alg.SwarmABC(population_size, max_iterations, model)
    swarm_abc.generate_initial_population()
    assert len(swarm_abc.elements) == population_size


@pytest.mark.parametrize("mandatory, optional", [
    (MandatorySettingsPSO(), OptionalSettingsPSO(inertia_setter=2, c_setter=1)),
    (MandatorySettingsPSO(), OptionalSettingsPSO(inertia_setter=3, c_setter=2)),
    (MandatorySettingsPSO(), OptionalSettingsPSO(inertia_setter=1, stop_inertia=0.5, c_setter=2))
])
def test_dynamic_parameters_pso(mandatory, optional):
    """
    Checks if the dynamic parameters of PSO are set correctly.
    :param mandatory: set of mandatory parameters.
    :param optional: set of optional parameters.
    """
    model = ModelProperties(optional.orbit_filepath, mandatory.number_of_measurements)
    swarm_pso = pso.SwarmPSO(mandatory.max_iterations, mandatory.population_size,
                            model, 1, 1, 1)
    swarm_pso.c_setter(200, optional.c_setter)
    swarm_pso.inertia_setter(200, optional.inertia_setter, mandatory.inertia, optional.stop_inertia)
    assert swarm_pso.c1 in [1.75, 0.25]
    assert swarm_pso.inertia in [0.65, 0.75, 0.96]


@pytest.mark.parametrize("mandatory, optional", [
    (MandatorySettingsABC(), OptionalSettingsABC(neighbourhood_type=1)),
    (MandatorySettingsABC(onlooker_phase_neighbours=6,
                          neighbours_pos_limits=150,
                          neighbours_vel_limits=0.01),
     OptionalSettingsABC(neighbourhood_type=1)),
])
def test_generating_neighbours_abc(mandatory, optional):
    """
    Checks if neighbours in ABC are generated correctly.
    :param mandatory: set of mandatory parameters.
    :param optional: set of optional parameters.
    :return: 
    """
    model = ModelProperties(optional.orbit_filepath, mandatory.number_of_measurements)
    swarm_abc = abc_alg.SwarmABC(mandatory.population_size, mandatory.max_iterations, model)
    food_abc = abc_alg.Food(6 * [0], swarm_abc, model)
    food_abc.generate_neighbours(mandatory.onlooker_phase_neighbours,
                                 [mandatory.neighbours_pos_limits, mandatory.neighbours_vel_limits],
                                 optional.neighbourhood_type, optional.dim_probability)
    assert len(food_abc.neighbours) == mandatory.onlooker_phase_neighbours
    diff_vector = [abs(np.array(food_abc.state) - np.array(food_abc.neighbours[i].state))
                   for i in range(mandatory.onlooker_phase_neighbours)]
    max_pos_diff = max(max(elem[:3]) for elem in diff_vector)
    max_vel_diff = max(max(elem[3:]) for elem in diff_vector)
    assert max_pos_diff <= mandatory.neighbours_pos_limits / 389703
    assert max_vel_diff <= mandatory.neighbours_vel_limits / (389703/382981)