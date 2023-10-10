import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Parametrii pentru distribuția exponentială
lambda1 = 4  # Pentru primul mecanic
lambda2 = 6  # Pentru al doilea mecanic

# Probabilitatea de a fi servit de primul mecanic
p_servire_primul_mecanic = 0.4

# Numărul total de clienți de servit
numar_clienti = 10000

# Generăm aleatoriu valori pentru a decide mecanicul care îl va servi pe fiecare client
alegeri_mecanici = np.random.choice([1, 2], numar_clienti, p=[p_servire_primul_mecanic, 1 - p_servire_primul_mecanic])

# Generăm valori pentru timpul de servire X în funcție de mecanicul ales
timp_de_servire = np.zeros(numar_clienti)

for i in range(numar_clienti):
    if alegeri_mecanici[i] == 1:
        timp_de_servire[i] = stats.expon(scale=1/lambda1).rvs()
    else:
        timp_de_servire[i] = stats.expon(scale=1/lambda2).rvs()

# Estimăm media și deviația standard a lui X
media_x = np.mean(timp_de_servire)
deviatia_standard_x = np.std(timp_de_servire)

print(f"Media lui X: {media_x}")
print(f"Deviația standard a lui X: {deviatia_standard_x}")

# Generăm un grafic al densității distribuției lui X
plt.hist(timp_de_servire, bins=50, density=True, alpha=0.6, color='0', label='Distribuția lui X')

plt.title('Densitatea distribuției lui X')
plt.xlabel('X (timp de servire)')
plt.ylabel('Densitate')
plt.legend()
plt.show()
