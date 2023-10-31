import pymc3 as pm
import numpy as np
import pandas as pd

# citirea datelor din fisier
nume_fisier = "trafic.csv"
df = pd.read_csv(nume_fisier)

# extragem coloana cu datele de trafic
traffic_data = df["traffic_values"].values

with pm.Model() as traffic_model:
    # parametrul global lambda ca fiind un parametru necunoscut pozitiv
    lambda_global = pm.Exponential("lambda_global", 1.0)

    # intervalele de timp cunoscute pentru crestere si descrestere
    ore_cresteri = [7, 16]
    ore_descresteri = [8, 19]

    # variabile pentru lambda  în fiecare minut
    lambdas = []

    for i in range(len(traffic_data)):
        if i // 60 in ore_cresteri:
            # daca suntem intr-un interval de crestere cunoscut, folosim alt parametru lambda
            lambda_i = pm.Exponential(f"lambda_{i}", 1.0)
            lambdas.append(lambda_i)
        elif i // 60 in ore_descresteri:
            # daca suntem intr-un interval de descrestere cunoscut, folosim alt parametru lambda
            lambda_i = pm.Exponential(f"lambda_{i}", 1.0)
            lambdas.append(lambda_i)
        else:
            # in afara acestor intervale, folosim lambda  global
            lambdas.append(lambda_global)

    # generam distributia poisson folosind lambda  pentru fiecare minut
    traffic = pm.Poisson("traffic", mu=lambdas, observed=traffic_data)

with traffic_model:
    trace = pm.sample(2000, tune=1000)

pm.summary(trace)

#determinarea capetelor celor 5 intervale de timp
quantiles = np.percentile(trace["lambda_global"], [2.5, 25, 50, 75, 97.5])

print("Capetele intervalelor de incredere de 95% pentru lambda  global:")
print("Interval 1:", quantiles[0], " - ", quantiles[1])
print("Interval 2:", quantiles[1], " - ", quantiles[2])
print("Interval 3:", quantiles[2], " - ", quantiles[3])
print("Interval 4:", quantiles[3], " - ", quantiles[4])

#determinarea celor mai probabile valori ale var lambda  global în fiecare interval
print("Cele mai probabile valori ale lambda global in fiecare interval:")
for i in range(4):
    interval_inferior = quantiles[i]
    interval_superior = quantiles[i + 1]
    interval_medie = np.mean(trace["lambda_global"][(trace["lambda_global"] >= interval_inferior) & (trace["lambda_global"] <= interval_superior)])
    print(f"Interval {i + 1}: {interval_medie}")

#Bonus1
# import matplotlib.pyplot as plt
#
# #setam seed-ul pentru reproducibilitate
# np.random.seed(0)
#
# # Parametrul de forma alpha
# alpha = 3
#
# #nr de timpi de asteptare medii pe care dorim sa ii generam
# num_samples = 100
#
# #eșantionul de timpi de asteptare medii folosind distributia gamma
# wait_times = np.random.gamma(shape=alpha, scale=1, size=num_samples)
#
# # afisam histograma timilor de asteptare medii
# plt.hist(wait_times, bins=20, density=True, alpha=0.6, color='g', label='Timpi de asteptare medii')
# plt.xlabel('Timp de asteptare mediu')
# plt.ylabel('Densitate')
# plt.title(f'Histograma timpilor de așteptare medii cu alpha = {alpha}')
# plt.legend()
# plt.show()
