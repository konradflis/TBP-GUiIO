import pandas as pd
from sources.data_structures import MandatorySettingsFA, OptionalSettingsFA
from sources.firefly_alg import firefly_alg
import itertools


def run_param_sweep_and_save(num_runs, param_grid, excel_filename="firefly_results_second.xlsx"):
    keys = list(param_grid.keys())
    combinations = list(itertools.product(*(param_grid[key] for key in keys)))
    ammount_comb = len(combinations)

    all_results = []
    count_test = 0

    for combo in combinations:
        mandatory_params = MandatorySettingsFA()
        optional_params = OptionalSettingsFA()

        # Ustawiamy stałe parametry
        mandatory_params.number_of_measurements = 25
        mandatory_params.alpha_initial = 0.5
        mandatory_params.gamma = 1
        mandatory_params.beta0 = 1

        print("Current test:", count_test + 1, "out of", ammount_comb, "combinations")

        param_settings = dict(zip(keys, combo))
        for key, value in param_settings.items():
            if hasattr(mandatory_params, key):
                setattr(mandatory_params, key, value)
            elif hasattr(optional_params, key):
                setattr(optional_params, key, value)
            else:
                print(f"Warning: {key} not found in settings!")

        last_iteration_scores = []
        best_scores = []
        best_iterations = []
        all_scores_per_run = []

        for i in range(num_runs):
            _, _, _, _, _, best_scores_vec = firefly_alg(mandatory_params, optional_params)

            last_iteration_scores.append(best_scores_vec[-1])
            best_score = min(best_scores_vec)
            best_scores.append(best_score)
            best_iterations.append(best_scores_vec.index(best_score))

            all_scores_per_run.append(", ".join(f"{x:.5f}" for x in best_scores_vec))

        result_entry = {
            # Agregowane wyniki
            "Mean Final Score (Last Iteration)": sum(last_iteration_scores) / num_runs,
            "Mean Final Best Iteration": int(round(sum(best_iterations) / num_runs)),
            "Min Best Score": min(best_scores),
            "All Scores Per Iteration": " | ".join(all_scores_per_run),

            # Parametry mandatory
            "Max Iterations": mandatory_params.max_iterations,
            "Population Size": mandatory_params.population_size,
            "Measurements": mandatory_params.number_of_measurements,
            "Alpha Initial": mandatory_params.alpha_initial,
            "Gamma": mandatory_params.gamma,
            "Beta0": mandatory_params.beta0,

            # Parametry optional
            "Alpha Decay": optional_params.alpha_decay,
            "Attractiveness Function": optional_params.attractiveness_function,
            "Distance Metric": optional_params.distance_metric,
            "Compare Type": optional_params.compare_type,
            "Movement Type": optional_params.movement_type,
            "Orbit File": str(optional_params.orbit_filepath),
        }

        all_results.append(result_entry)
        count_test += 1

    df = pd.DataFrame(all_results)

    try:
        existing_df = pd.read_excel(excel_filename)
        df = pd.concat([existing_df, df], ignore_index=True)
    except FileNotFoundError:
        pass

    df.to_excel(excel_filename, index=False)
    print(f"Wyniki zapisane do {excel_filename}")


# Tylko te trzy parametry będą się zmieniać
param_grid = {
    "max_iterations": [10, 50, 100], #
    "population_size": [10, 25, 40], #
    "compare_type": ['all-all','all-all-no-duplicates', 'by-pairs']
}

run_param_sweep_and_save(num_runs=5, param_grid=param_grid)
