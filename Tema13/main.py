import numpy as np
import arviz as az
from matplotlib import pyplot as plt
import sys
sys.stdout.reconfigure(encoding='utf-8')

#ex1

centered_data = az.load_arviz_data("centered_eight")

#informatiile despre modelul centrat
print("Modelul Centrat:")
print("Numarul de lanturi:", centered_data.posterior.chain.size)
print("Marimea totala a esantionului generat:", centered_data.posterior.draw.size)

# distribuția a posteriori pentru modelul centrat
az.plot_posterior(centered_data)


non_centered_data = az.load_arviz_data("non_centered_eight")

#informațiile despre modelul necentrat
print("\nModelul Necentrat:")
print("Nr de lanturi:", non_centered_data.posterior.chain.size)
print("Marimea totala a esantionului generat:", non_centered_data.posterior.draw.size)

#  distributia a posteriori pentru modelul necentrat
az.plot_posterior(non_centered_data)

# Afișați graficele
plt.show()


#ex2
centered_data = az.load_arviz_data("centered_eight")
non_centered_data = az.load_arviz_data("non_centered_eight")

# Afisam Rhat pentru parametrii mu si tau pentru modelul centrat
print("Rhat pentru modelul centrat:")
print(az.summary(centered_data, var_names=["mu", "tau"], r_hat=True))

#autocorelatia pentru parametrii mu si tau pentru modelul centrat
az.plot_autocorr(centered_data, var_names=["mu", "tau"])

# Afisam Rhat pentru parametrii mu si tau pentru modelul necentrat
print("\nRhat pentru modelul necentrat:")
print(az.summary(non_centered_data, var_names=["mu", "tau"], r_hat=True))

#  autocorelația pentru parametrii mu și tau pentru modelul necentrat
az.plot_autocorr(non_centered_data, var_names=["mu", "tau"])

# Afișați graficele
plt.show()


#ex3

centered_data = az.load_arviz_data("centered_eight")
non_centered_data = az.load_arviz_data("non_centered_eight")

# Numaram divergentele pentru modelele centrat si necentrat
divergences_centered = centered_data.sample_stats.diverging.sum()
divergences_non_centered = non_centered_data.sample_stats.diverging.sum()

# Afisam numarul de divergente pentru fiecare model
print("Nr de divergente pentru modelul centrat:", divergences_centered)
print("Nr de divergente pentru modelul necentrat:", divergences_non_centered)

#distributia divergentelor in spatiul parametrilor f
az.plot_pair(centered_data, var_names=["mu", "tau"], divergences=True)
plt.suptitle("Divergente in modelul centrat")
plt.show()

az.plot_pair(non_centered_data, var_names=["mu", "tau"], divergences=True)
plt.suptitle("Divergente in modelul necentrat")
plt.show()