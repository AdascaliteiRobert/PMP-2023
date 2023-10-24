import numpy as np


lambd = 20
mean_Y = 2  # minute
stddev_Y = 0.5  # minute
alpha = 0.1  # Rata de servire (1/minut)

num_clients = np.random.poisson(lambd)

times_Y = np.random.normal(mean_Y, stddev_Y, num_clients)
times_Z = np.random.exponential(alpha, num_clients)
times_total = times_Y + times_Z


print("Rezultate pentru alpha =", alpha)
print(f"Numarul de clienti: {num_clients}")
print(f"Timpul de plasare si plata a comenzilor: {times_Y}")
print(f"Timpul de pregatire la statia de gatit: {times_Z}")
print(f"Timpul total de servire pentru fiecare comanda: {times_total}")

max_total_time = 15  # minute
required_probability = 0.95

max_alpha = 0
waiting_time = 0

while alpha < 100:
    num_clients = np.random.poisson(lambd)
    times_Y = np.random.normal(mean_Y, stddev_Y, num_clients)
    times_Z = np.random.exponential(1 / alpha, num_clients)
    times_total = times_Y + times_Z

    served_in_time = np.sum(times_total < max_total_time)
    probability = served_in_time / num_clients

    if probability >= required_probability:
        max_alpha = alpha
        break
    alpha += 0.1

if max_alpha > 0:
    waiting_time = (mean_Y+alpha)/2

print(f"Valoarea maximă pentru alpha este: {max_alpha} (1/minut)")
print(f"Timpul mediu de așteptare pentru a fi servit unui client: {waiting_time:.2f} minute")
