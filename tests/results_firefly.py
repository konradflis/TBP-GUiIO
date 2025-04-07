import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Wczytanie pliku z wynikami
file_path = "firefly_results_second.xlsx"  # Zmienna na ścieżkę do pliku
df = pd.read_excel(file_path)

# Przetwarzamy dane - grupowanie po compare_type
compare_types = df['Compare Type'].unique()

# Ustalamy liczbę subplotów w zależności od liczby metod compare_type
fig, axes = plt.subplots(len(compare_types), 1, figsize=(10, 5 * len(compare_types)))
if len(compare_types) == 1:
    axes = [axes]  # Ujednolicenie do listy, jeśli tylko jeden subplot

# Ustawienie ogólnego stylu
sns.set(style="whitegrid")

# Pętla po różnych metodach 'compare_type'
for i, compare_type in enumerate(compare_types):
    ax = axes[i]

    # Filtrowanie wyników dla konkretnego compare_type
    compare_data = df[df['Compare Type'] == compare_type]

    # Rysowanie krzywych konwergencji dla wszystkich testów w tej metodzie
    for _, row in compare_data.iterrows():
        best_scores_per_iteration = list(map(float, row['All Scores Per Iteration'].split(", ")))
        ax.plot(best_scores_per_iteration, label=f"Test {row.name + 1}")

    ax.set_title(f"Convergence Curves for {compare_type}")
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Best Score')
    ax.legend(loc='upper right')

plt.tight_layout()  # Dopasowanie układu
plt.show()

# Zapisanie wykresu do pliku
output_path = "convergence_curves.png"
fig.savefig(output_path)

print(f"Wykresy zostały zapisane w {output_path}")
