# pylint: disable=R0902, R0903, R0913, R0917
"""
Classes defining data structures for mandatory and optional parameters,
structures for translating dynamic widgets.
"""
from pathlib import Path

class MandatorySettingsPSO:
    """
    Mandatory fields for PSO algorithm.
    """
    def __init__(self,
                 max_iterations=10,
                 population_size=10,
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
                 max_iterations=10,
                 population_size=10,
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
                 number_of_multistarts=5,
                 orbit_filepath=Path(__file__).resolve().parent.parent / "orbits" / "L2_7days.txt"
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
                 orbit_filepath=Path(__file__).resolve().parent.parent / "orbits" / "L2_7days.txt"
                 ):
        self.inactive_cycles_setter = inactive_cycles_setter
        self.probability_distribution_setter = probability_distribution_setter
        self.generating_method = generating_method
        self.neighbourhood_type = neighbourhood_type
        self.neigh_percent = neigh_percent
        self.dim_probability = dim_probability
        self.orbit_filepath = orbit_filepath


class MandatorySettingsGEN:
    """
    Mandatory fields for GEN algorithm.
    """
    def __init__(self,
                 population_size=10,
                 number_of_measurements=35,
                 max_generations=5,
                 mutation_rate=0.01,
                 crossover_rate=0.7,
                 ):
        self.population_size = population_size
        self.number_of_measurements = number_of_measurements
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

class OptionalSettingsGEN:
    """
    Optional fields for GEN algorithm.
    """
    def __init__(self,
                 orbit_filepath=Path(__file__).resolve().parent.parent / "orbits" / "L2_7days.txt",
                 select_parent_opt=True, tournament_size=None, mutate_opt=True):
        self.orbit_filepath = orbit_filepath
        self.select_parent_opt = select_parent_opt
        self.tournament_size = tournament_size
        self.mutate_opt = mutate_opt

class PlotSettings:
    """
    Settings for plotting trajectories.
    """
    def __init__(self,
                 density=0,
                 periods=1):
        self.density = density
        self.periods = periods

class Translations:
    """
    Class defining a dictionary of translations for widgets.
    """
    def __init__(self):
        self.dictionary = {
            "Table": {
                "PL": ["cel", "wynik", "|różnica|", "|odległość|", "fun. celu"],
                "EN": ["objective", "result", "|diff|", "|distance|", "obj. fun. val."]
                },
            "Plot": {
                "gridOrbit" : {
                    "PL" : ["oryginalna orbita", "otrzymana orbita"],
                    "EN" : ["original orbit", "propagated orbit"]
                },
                "gridPosition" : {
                    "PL" : ["położenia początkowe", "położenia końcowe"],
                    "EN" : ["initial positions", "final positions"]
                },
                "gridVelocity" : {
                    "PL" : ["prędkości początkowe", "prędkości końcowe"],
                    "EN" : ["initial velocities", "final velocities"]
                },
                "gridError" : {
                    "PL" : ["Numer iteracji", "Wartość błędu/funkcji celu [LU]"],
                    "EN" : ["Iteration number", "Objective function value [LU]"]
                }
            }
        }

    def get_translation(self, widget_type, language, plot_type=None):
        """
        Get a specific translation for the given widget type, algorithm, and key.
        """
        try:
            if plot_type:
                return self.dictionary[widget_type][plot_type][language]
            return self.dictionary[widget_type][language]
        except KeyError as not_found_error:
            raise KeyError(
                f"No translation for language={language}, "
                f"widget_type={widget_type}, plot type={plot_type}"
            ) from not_found_error


class ValidatedElement:
    """
    Class defining the properties of validated element - its expected type, min and max values.
    """
    def __init__(self,
                 expected_type,
                 min_value,
                 max_value,
                 mapped_attribute):
        self.expected_type = expected_type
        self.min_value = min_value
        self.max_value = max_value
        self.mapped_attribute = mapped_attribute


class Validations:
    """
    Class defining a dictionary of validations for user inputs.
    """
    def __init__(self):
        self.dictionary = {
            "PSOmaxIterations": ValidatedElement(int, 1, 999,
                                                 "mandatory_pso.max_iterations"),
            "PSOpopulationSize": ValidatedElement(int, 1, 999,
                                                  "mandatory_pso.population_size"),
            "PSOnumberOfMeasurements": ValidatedElement(int, 1, 250,
                                                        "mandatory_pso.number_of_measurements"),
            "PSOinertia": ValidatedElement(float, 0, 2,
                                           "mandatory_pso.inertia"),
            "PSOc1": ValidatedElement(float, 0, 4,
                                      "mandatory_pso.c1"),
            "PSOc2": ValidatedElement(float, 0, 4,
                                      "mandatory_pso.c2"),
            "PSOstopInertia": ValidatedElement(float, 0, 2,
                                               "optional_pso.stop_inertia"),
            "PSO2maxIterations1": ValidatedElement(int, 1, 999,
                                                   "mandatory_pso2_1.max_iterations"),
            "PSO2maxIterations2": ValidatedElement(int, 1, 999,
                                                   "mandatory_pso2_2.max_iterations"),
            "PSO2populationSize1": ValidatedElement(int, 1, 999,
                                                    "mandatory_pso2_1.population_size"),
            "PSO2populationSize2": ValidatedElement(int, 1, 999,
                                                    "mandatory_pso2_2.population_size"),
            "PSO2numberOfMeasurements1": ValidatedElement(int, 1, 250,
                                                    "mandatory_pso2_1.number_of_measurements"),
            "PSO2numberOfMeasurements2": ValidatedElement(int, 1, 250,
                                                    "mandatory_pso2_2.number_of_measurements"),
            "PSO2inertia1": ValidatedElement(float, 0, 2,
                                             "mandatory_pso2_1.inertia"),
            "PSO2inertia2": ValidatedElement(float, 0, 2,
                                             "mandatory_pso2_2.inertia"),
            "PSO2c11": ValidatedElement(float, 0, 4,
                                        "mandatory_pso2_1.c1"),
            "PSO2c21": ValidatedElement(float, 0, 4,
                                        "mandatory_pso2_1.c2"),
            "PSO2c12": ValidatedElement(float, 0, 4,
                                        "mandatory_pso2_2.c1"),
            "PSO2c22": ValidatedElement(float, 0, 4,
                                        "mandatory_pso2_2.c2"),
            "PSO2multistart2": ValidatedElement(int, 1, 25,
                                                "optional_pso2_2.number_of_multistarts"),
            "PSO21stopInertia": ValidatedElement(float, 0, 2,
                                                 "optional_pso2_1.stop_inertia"),
            "PSO22stopInertia": ValidatedElement(float, 0, 2,
                                                 "optional_pso2_2.stop_inertia"),
            "ABCmaxIterations": ValidatedElement(int, 1, 999,
                                                 "mandatory_abc.max_iterations"),
            "ABCpopulationSize": ValidatedElement(int, 1, 999,
                                                  "mandatory_abc.population_size"),
            "ABCnumberOfMeasurements": ValidatedElement(int, 1, 250,
                                                        "mandatory_abc.number_of_measurements"),
            "ABCneighboursFirst": ValidatedElement(int, 1, 20,
                                                    "mandatory_abc.employee_phase_neighbours"),
            "ABCneighboursSecond": ValidatedElement(int, 1, 20,
                                                    "mandatory_abc.onlooker_phase_neighbours"),
            "ABCplaceLimits": ValidatedElement(float, 1, 250,
                                                       "mandatory_abc.neighbours_pos_limits"),
            "ABCvelocityLimit": ValidatedElement(float, 0, 0.1,
                                                       "mandatory_abc.neighbours_vel_limits"),
            "ABCinactiveCycles": ValidatedElement(int, 1, 99,
                                                       "mandatory_abc.inactive_cycles_limit"),
            "ABCdimProbability": ValidatedElement(float, 0, 1,
                                                  "optional_abc.dim_probability"),
            "ABCneighPercent": ValidatedElement(float, 0, 1,
                                                "optional_abc.neigh_percent"),
            "multiplePeriods": ValidatedElement(int, 0, 99,
                                                "optional_abc.multiple_periods"),
            "GEN_pop_size": ValidatedElement(int, 1, 999,
                                                "mandatory_gen.population_size"),
            "GEN_max_gen": ValidatedElement(int, 1, 999,
                                                "mandatory_gen.max_generations"),
            "GEN_mut_rate": ValidatedElement(float, 0, 1,
                                                "mandatory_gen.mutation_rate"),
            "GEN_cross_rate": ValidatedElement(float, 0, 1,
                                                "mandatory_gen.crossover_rate"),
            "GENnumberOfMeasurements": ValidatedElement(int, 1, 250,
                                                        "mandatory_gen.number_of_measurements")
        }
