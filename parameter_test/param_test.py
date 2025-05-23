import pandas as pd
from sources.genetic_alg import GeneticAlgorithm
from sources.data_structures import MandatorySettingsGEN, OptionalSettingsGEN
from sources.common_elements import PropagatedElement, ModelProperties
from pathlib import Path
import json

# Parametry algorytmu
population_size = 500
max_generations = 10
mutation_rate = 0.01
crossover_rate = 0.7
number_of_measurements  = 5

# parametry opcjonalne
orbit_filepath=Path(__file__).resolve().parent.parent / "orbits" / "L2_7days.txt" #wybór orbity
select_parent_opt = False       # wybór opcji selekcji (True = turniej, False = ruletka)
tournament_size = None          # jeśli metoda turniejowa wybierz rozmiar turnieju
mutate_opt = False              # wybór opcji mutacji (False = stałe przesunięcie, True=losowe przsunięcie)
both_methods = True             # True = wybór mieszanych metod krzyżowania, False = wybór konkretnej metody
single_point = True             # jeżeli konkretna metoda, krzyżowanie: True = jednopunktowe, False = jednorodne

# Ścieżka do pliku CSV
csv_filename = Path(__file__).parent / 'data' / 'test9' / 'result.csv'
csv_filename.parent.mkdir(parents=True, exist_ok=True)

# Uruchomienie algorytmu 10 razy i zapisanie wyników
mandatory= MandatorySettingsGEN(
    population_size = population_size,
    number_of_measurements = number_of_measurements ,
    max_generations = max_generations,
    mutation_rate = mutation_rate,
    crossover_rate = crossover_rate,
)
optional = OptionalSettingsGEN(
    orbit_filepath = orbit_filepath,
    select_parent_opt = select_parent_opt,
    tournament_size = tournament_size,
    mutate_opt = mutate_opt,
    both_methods = both_methods,
    single_point = both_methods
)

results = []
for i in range(10):
    ga = GeneticAlgorithm(
        mandatory = mandatory,
        optional=optional
    )
    ga.run()
    best_individual = min(ga.population.individuals, key=lambda ind: ind.score)
    best_state = [float(val) for val in best_individual.state]

    results.append([i + 1, best_individual.score, best_state])

# Zapisanie wyników do DataFrame i potem do pliku CSV
df = pd.DataFrame(results, columns=["Run", "Best Score", "Best State"])

# Zapisanie do CSV (z nagłówkami)
df.to_csv(csv_filename, index=False)
print(f"Wynik zapisany {csv_filename}")

json_filename = Path(__file__).parent / 'data' / 'test9' / 'params.json'
json_filename.parent.mkdir(parents=True, exist_ok=True)

params = {
    'population_size': population_size,
    'max_generations': max_generations,
    'mutation_rate': mutation_rate,
    'crossover_rate': crossover_rate,
    'tournament_size': tournament_size
}
with open(json_filename, 'w') as json_file:
    json.dump(params, json_file, indent=4)

print(f"Parameters saved to {json_filename}")


