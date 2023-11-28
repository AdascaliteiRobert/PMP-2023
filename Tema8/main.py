import pymc as pm
import pandas as pd
import numpy as np
import arviz as az


data = pd.read_csv('Prices.csv')

price = data['Price']
processor_speed = data['Speed']
hard_drive_size = np.log(data['HardDrive'])


with pm.Model() as model:
    # distributii a priori
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta1 = pm.Normal('beta1', mu=0, sigma=10)
    beta2 = pm.Normal('beta2', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=1)

    # regresiq
    mu = alpha + beta1 * processor_speed + beta2 * hard_drive_size

    likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=price)

    # distributia a posteriori
    trace = pm.sample(2000, tune=1000, cores=1)  # Adjustează numărul de eșantioane și tunare după nevoie

# estimarile HDI pentru beta1 , beta2
hdi_summary = az.summary(trace, hdi_prob=0.95)
hdi_beta1 = hdi_summary.loc['beta1', ['hdi_2.5%', 'hdi_97.5%']].values
hdi_beta2 = hdi_summary.loc['beta2', ['hdi_2.5%', 'hdi_97.5%']].values

print(f'Estimările HDI pentru beta1: {hdi_beta1}')
print(f'Estimările HDI pentru beta2: {hdi_beta2}')

#subpunctul 3

# Evaluarea utilitagii predictorilor
# in acest context, putem evalua importanta predictorilor pe baza magnitudinii si a faptului  dacq intervalele HDI contin zero.
#daca intervalul contine zero, atunci coeficientul asociat nu este semnificativ diferit de zero.

#subpunctul 4

# # simulam pretul de vanzare asteptat pentru un computer cu frecvența de 33 mhz si hard disk de 540 mb
# processor_speed_new = 33
# hard_drive_size_new = np.log(540)  # logaritmul natural al marimii hard diskului de 540 MB
#
# # simulam 5000 de extrageri din distributia a posteriori
# simulated_prices = pm.sample_posterior_predictive(trace, samples=5000, model=model, random_seed=42)
#
# # distributia preturilor simulate
# simulated_prices_dist = simulated_prices['likelihood']
#
# # pretul de vanzare asteptat pentru fiecare extragere simulata
# expected_prices = simulated_prices_dist.mean(axis=0)
#
# #intervalul HDI pentru prețul de vanzare asteptat
# hdi_expected_prices = az.hdi(expected_prices, hdi_prob=0.9)
#
# print(f'Intervalul HDI pentru pretul de vanzare asteptat (90%): {hdi_expected_prices}')

# subpunctul 5

# #simulam 5000 de extrageri din distributia predictiva a posteriori
# simulated_prices_predictive = pm.sample_posterior_predictive(trace, samples=5000, model=model, random_seed=42)
#
# #distribuția preturilor simulate
# simulated_prices_dist_predictive = simulated_prices_predictive['likelihood']
#
# # intervalul HDI pentru distributia predictiva
# hdi_predictive = az.hdi(simulated_prices_dist_predictive, hdi_prob=0.9)
#
# print(f'Intervalul HDI pentru pretul de vanzare predictiv (90%): {hdi_predictive}')

#bonnus

# Daca intervalul de incredere HDI pentru interceptare (alpha) nu contine zero, atunci exista o probabilitate semnificativa
# intrucat faptul ca producatorul este premium atunci  prețul computerelor este afectat.
# In caz contrar, daca intervalul contine zero, nu putem concluziona ca exist o influenta semnificativa a variabilei
# care indica producatorul premium asupra pretului.
#Este  posibil caa si alți factori sa joace un rol semnificativ in determinarea pretului  iar modelul de regresie
# ar putea sa nu captureze toate aspectele complexe ale relatiei.






