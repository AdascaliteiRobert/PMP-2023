# import pandas as pd
# import matplotlib.pyplot as plt
#
# # incarcam setul de date într-un DataFrame
# df = pd.read_csv('auto-mpg.csv')
#
# verificam structura datelor
# print(df.head())
#
#
# print(df.info())
#
#
# # Exemplu: Eliminarea randurilor cu valori lipsa
# df = df.dropna()
#
# # Trasez graficul
# plt.scatter(df['horsepower'], df['mpg'])
# plt.title('Relatia dintre cai putere si mpg')
# plt.xlabel('Cai Putere (horsepower)')
# plt.ylabel('Mile pe Galon (mpg)')
# plt.show()

#Punctul b)
# import pymc as pm
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Citim datele din fișierul CSV
# data = pd.read_csv('auto-mpg.csv')
#
# # Selectam doar coloanele de interes (CP și mpg)
# data = data[['horsepower', 'mpg']]
#
# # Eliminam randurile care conțin valori lipsa
# data = data.dropna()
#
# # Convertim coloana 'horsepower' la tip numeric
# data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')
#
# # Modelam relatia dintre CP si mpg folosind un model liniar Bayesian
# with pm.Model() as linear_model:
#     # Parametrii modelului
#     alpha = pm.Normal('alpha', mu=0, sigma=10)
#     beta = pm.Normal('beta', mu=0, sigma=10)
#
#     # Modelul liniar
#     mu = alpha + beta * data['horsepower']
#
#     # Distribuția așteptată a observațiilor
#     mpg = pm.Normal('mpg', mu=mu, sigma=1, observed=data['mpg'])
#
#     # Specificare manuals a valorilor initiale pentru parametrii
#     start = {'alpha': 0, 'beta': 0}  # Ajusteaza aceste valori dupa nevoie
#
# # Inferenta Bayesiana folosind Metropolis-Hastings
# with linear_model:
#     trace = pm.sample(1000, tune=1000, cores=1, start=start)
#
# # Analiza rezultatelor
# pm.summary(trace).round(2)
#
# # Afisam rezultatele
# pm.traceplot(trace)
# plt.show()

#Punctul c)

# import pandas as pd
# import numpy as np
# import pymc as pm
# import matplotlib.pyplot as plt

# df = pd.read_csv('auto-mpg.csv')

# print(df.head())
#
# # Convertiti coloana 'horsepower' in valori numerice
# df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
#
# X = df['horsepower'].values
# y = df['mpg'].values
#
# mask = ~np.isnan(X)
# X = X[mask]
# y = y[mask]
#
# # Modelul bayesian
# with pm.Model() as bayesian_model:
#     # Parametrii priori pentru intercept (alpha) si coeficientul de panta (beta)
#     alpha = pm.Normal('alpha', mu=0, sigma=10)
#     beta = pm.Normal('beta', mu=0, sigma=10)
#
#     # Modelul liniar
#     mu = alpha + beta * X
#
#     # Precizia (inversul variabilitatii) a distributiei normale pentru reziduurile
#     sigma = pm.HalfNormal('sigma', sigma=1)
#
#     # Likelihood (likelihood este format din distributia normala pentru datele observate)
#     likelihood = pm.Normal('mpg', mu=mu, sigma=sigma, observed=y)
#
#     # Inferenta bayesiana
#     trace = pm.sample(2000, tune=1000, cores=2)
#
# pm.summary(trace).round(2)
#
# # Afiaare distributii posterioare
# pm.traceplot(trace)
# plt.show()

#Punctul d)

# import pandas as pd
# import numpy as np
# import pymc as pm
# import arviz as az
# import matplotlib.pyplot as plt
#
# df = pd.read_csv('auto-mpg.csv')
#
# print(df.head())
#
# df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
#
# df_cleaned = df[['horsepower', 'mpg']].dropna()
#
# # Pregatim datele pentru modelul bayesian
# X = df_cleaned['horsepower'].values
# y = df_cleaned['mpg'].values
#
# # Modelul bayesian
# with pm.Model() as bayesian_model:
#     # Parametrii priori pentru intercept (alpha) si coeficientul de panta (beta)
#     alpha = pm.Normal('alpha', mu=0, sigma=10)
#     beta = pm.Normal('beta', mu=0, sigma=10)
#
#     # Modelul liniar
#     mu = alpha + beta * X
#
#     # Precizia (inversul variabilitatii) a distributiei normale pentru reziduurile
#     sigma = pm.HalfNormal('sigma', sigma=1)
#
#     # Likelihood (likelihood este format din distributia normala pentru datele observate)
#     likelihood = pm.Normal('mpg', mu=mu, sigma=sigma, observed=y)
#
#     # Inferenta bayesiana
#     trace = pm.sample(2000, tune=1000, cores=2)
#
# # Afisare rezultate
# print(az.summary(trace).round(2))
#
# # Afisare distributii posterioare
# az.plot_trace(trace)
#
# # Relatia dintre CP si mpg cu regiunea 95%HDI pentru distributia predictiva a posteriori
# plt.scatter(X, y, label='Date observate')
# pm.plot_posterior_predictive_glm(trace, samples=100, eval=np.linspace(X.min(), X.max(), 100), color='red', alpha=0.2, label='Distribuția predictivă a posteriori')
# plt.xlabel('Cai putere (CP)')
# plt.ylabel('Mile pe galon (mpg)')
# plt.legend()
# plt.show()
