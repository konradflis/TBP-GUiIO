# pylint: disable=R0902, R0903, R0913, R0917
"""
Classes defining data structures for mandatory and optional parameters.
"""

class MandatorySettingsPSO:
    """
    Mandatory fields for PSO algorithm.
    """
    def __init__(self,
                 max_iterations=100,
                 population_size=100,
                 number_of_measurements=35,
                 inertia=0.9,
                 c1=1.49,
                 c2=1.49,
                 ):
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.number_of_measurements = number_of_measurements
        self.inertia = inertia
        self.c1 = c1
        self.c2 = c2


class MandatorySettingsABC:
    """
    Mandatory fields for ABC algorithm.
    """
    def __init__(self,
                 max_iterations=100,
                 population_size=50,
                 number_of_measurements=35,
                 employee_phase_neighbours=2,
                 onlooker_phase_neighbours=2,
                 neighbours_pos_limits=5,
                 neighbours_vel_limits=0.005,
                 inactive_cycles_limit=5
                 ):
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.number_of_measurements = number_of_measurements
        self.employee_phase_neighbours = employee_phase_neighbours
        self.onlooker_phase_neighbours = onlooker_phase_neighbours
        self.neighbours_pos_limits = neighbours_pos_limits
        self.neighbours_vel_limits = neighbours_vel_limits
        self.inactive_cycles_limit = inactive_cycles_limit


class OptionalSettingsPSO:
    """
    Optional fields for PSO algorithm.
    """
    def __init__(self,
                 inertia_setter=0,
                 stop_inertia=0.6,
                 c_setter=0,
                 if_best_velocity=0,
                 best_velocity=None,
                 multistart=0,
                 number_of_multistarts=20,
                 orbit_filepath="../orbits/L2_7days.txt"
                 ):
        self.inertia_setter = inertia_setter
        self.stop_inertia = stop_inertia
        self.c_setter = c_setter
        self.if_best_velocity = if_best_velocity
        self.best_velocity = best_velocity
        self.multistart = multistart
        self.number_of_multistarts = number_of_multistarts
        self.orbit_filepath = orbit_filepath


class OptionalSettingsABC:
    """
    Optional fields for ABC algorithm.
    """
    def __init__(self,
                 inactive_cycles_setter=0,
                 probability_distribution_setter=0,
                 generating_method=0,
                 neighbourhood_type=0,
                 neigh_percent=0.02,
                 dim_probability=0.5,
                 orbit_filepath="../orbits/L2_7days.txt"
                 ):
        self.inactive_cycles_setter = inactive_cycles_setter
        self.probability_distribution_setter = probability_distribution_setter
        self.generating_method = generating_method
        self.neighbourhood_type = neighbourhood_type
        self.neigh_percent = neigh_percent
        self.dim_probability = dim_probability
        self.orbit_filepath = orbit_filepath
