import pandas as pd
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt

# Citirea datelor
data = pd.read_csv('date.csv')

# Definirea modelului
with pm.Model() as model_p:
    pm.GLM.from_formula('y ~ x**5', data)
    trace = pm.sample(1000, tune=1000)

# Inferența și reprezentarea grafică
pm.plot_posterior(trace)
plt.show()


# Pentru sd=100
with pm.Model() as model_sd_100:
    pm.GLM.from_formula('y ~ x**5', data)
    trace_sd_100 = pm.sample(1000, tune=1000)

# Pentru sd=np.array([10, 0.1, 0.1, 0.1, 0.1])
with pm.Model() as model_sd_array:
    pm.GLM.from_formula('y ~ x**5', data)
    pm.Normal('beta', mu=0, sd=np.array([10, 0.1, 0.1, 0.1, 0.1]), shape=5)
    trace_sd_array = pm.sample(1000, tune=1000)

# Inferenta si reprezentarea grafica pentru sd=100
pm.plot_posterior(trace_sd_100)
plt.show()

# Inferenta si reprezentarea grafica pentru sd=np.array([10, 0.1, 0.1, 0.1, 0.1])
pm.plot_posterior(trace_sd_array)
plt.show()

# Generare de date cu 500 de puncte
np.random.seed(42)
data_500 = pd.DataFrame({'x': np.linspace(0, 10, 500)})
data_500['y'] = 2 * data_500['x']**5 + np.random.normal(0, 10, 500)

# Inferenta pentru 500 de puncte
with pm.Model() as model_500:
    pm.GLM.from_formula('y ~ x**5', data_500)
    trace_500 = pm.sample(1000, tune=1000)

# inferenta si reprezentarea graficaa pentru 500 de puncte
pm.plot_posterior(trace_500)
plt.show()

# Model cubic (order=3)
with pm.Model() as model_cubic:
    pm.GLM.from_formula('y ~ x**3', data)
    trace_cubic = pm.sample(1000, tune=1000)

# Calculam WAIC și LOO
waic_linear = pm.waic(trace)
waic_quadratic = pm.waic(trace_sd_100)
waic_cubic = pm.waic(trace_cubic)

loo_linear = pm.loo(trace)
loo_quadratic = pm.loo(trace_sd_100)
loo_cubic = pm.loo(trace_cubic)

# Reprzentarea grafica si comparatia
pm.compare({'linear': trace, 'quadratic': trace_sd_100, 'cubic': trace_cubic}, ic='waic')
plt.show()

pm.compare({'linear': trace, 'quadratic': trace_sd_100, 'cubic': trace_cubic}, ic='loo')
plt.show()
