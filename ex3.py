import numpy as np
import matplotlib.pyplot as plt

# Numărul de experimente
num_experiments = 100

# Lansarea unei monezi "ss" (ambele monezi arată stemă)
p_ss = 0.3 * 0.3

# Lansarea unei monezi "sb" (prima arată stemă, a doua arată ban)
p_sb = 0.3 * 0.7

# Lansarea unei monezi "bs" (prima arată ban, a doua arată stemă)
p_bs = 0.7 * 0.3
# Lansarea unei monezi "bb" (ambele monezi arată ban)
p_bb = 0.7 * 0.7

# Lista pentru a stoca rezultatele
results = []

# Generează rezultatele pentru 100 de experimente
for _ in range(num_experiments):
    experiment = np.random.choice(["ss", "sb", "bs", "bb"], size=10, p=[p_ss, p_sb, p_bs, p_bb])
    results.append(experiment)

# Calculează frecvența fiecărui rezultat posibil în cele 100 de experimente
counts = {"ss": 0, "sb": 0, "bs": 0, "bb": 0}

for experiment in results:
    for outcome in experiment:
        counts[outcome] += 1

# Creează un grafic pentru distribuția rezultatelor
outcomes = list(counts.keys())
frequencies = [counts[outcome] / (num_experiments * 10) for outcome in outcomes]

plt.bar(outcomes, frequencies)
plt.xlabel("Rezultat")
plt.ylabel("Probabilitate")
plt.title("Distribuția rezultatelor în 100 de experimente")
plt.show()