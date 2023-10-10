import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Parametrii pentru distribuțiile gamma ale serverelor
alpha1, lambda1 = 4, 1/3  # Primul server: Γ(4, 3)
alpha2, lambda2 = 4, 1/2  # Al doilea server: Γ(4, 2)
alpha3, lambda3 = 5, 1/2  # Al treilea server: Γ(5, 2)
alpha4, lambda4 = 5, 1/3  # Al patrulea server: Γ(5, 3)

# Probabilitatea de direcționare către fiecare server
probabilitate_server1 = 0.25
probabilitate_server2 = 0.25
probabilitate_server3 = 0.30
probabilitate_server4 = 1 - probabilitate_server1 - probabilitate_server2 - probabilitate_server3

# Numărul de eșantioane pentru simulare
numar_eșantioane = 10000


# Funcție pentru calculul timpului necesar pentru servirea unui client (X)
def timp_servire_client():
    server_ales = np.random.choice([1, 2, 3, 4], p=[probabilitate_server1, probabilitate_server2, probabilitate_server3, probabilitate_server4])
    if server_ales == 1:
        return stats.gamma.rvs(alpha1, scale=1/lambda1) + stats.expon.rvs(scale=1/4)
    elif server_ales == 2:
        return stats.gamma.rvs(alpha2, scale=1/lambda2) + stats.expon.rvs(scale=1/4)
    elif server_ales == 3:
        return stats.gamma.rvs(alpha3, scale=1/lambda3) + stats.expon.rvs(scale=1/4)
    else:
        return stats.gamma.rvs(alpha4, scale=1/lambda4) + stats.expon.rvs(scale=1/4)

# Simulăm timpurile de servire pentru toți clienții
timpuri_servire = np.array([timp_servire_client() for _ in range(numar_eșantioane)])

# Calculăm probabilitatea ca timpul de servire X să fie mai mare de 3 milisecunde
probabilitate_X_mai_mare_de_3 = np.sum(timpuri_servire > 3) / numar_eșantioane

print(f"Probabilitatea ca X > 3 ms: {probabilitate_X_mai_mare_de_3}")

# Generăm un grafic al densității distribuției lui X
plt.hist(timpuri_servire, bins=50, density=True, alpha=0.6, color='0', label='Distribuția lui X')
plt.title('Densitatea distribuției lui X')
plt.xlabel('X (timp de servire)')
plt.ylabel('Densitate')
plt.legend()
plt.show()