import random
import numpy as np
from pgmpy.models import BayesianModel
from pgmpy.estimators import ParameterEstimator, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# #Subiectul 1

# #punctul 1
def arunca_moneda_masluita():
    # Returneaza 1 cu probabilitatea 2/3 si 0 cu probabilitatea 1/3
    return 1 if random.random() < 2/3 else 0

def arunca_moneda_normala():
    # Returneaza 0 sau 1 cu probabilitatea 1/2
    return random.choice([0, 1])

def simuleaza_joc():
    # Simuleaza o runda de joc, returnand castigatorul
    castigator_runda_1 = arunca_moneda_normala()
    castigator_runda_2 = arunca_moneda_masluita()

    return castigator_runda_1 if castigator_runda_1 >= castigator_runda_2 else 1

def simulare_multipla(numar_jocuri):
    # Simuleaza de mai multe ori si returneaza procentajele de castig pentru jucatorii j0 si j1
    castiguri_j0 = 0
    castiguri_j1 = 0

    for _ in range(numar_jocuri):
        castigator = simuleaza_joc()
        if castigator == 0:
            castiguri_j0 += 1
        else:
            castiguri_j1 += 1

    procentaj_j0 = (castiguri_j0 / numar_jocuri) * 100
    procentaj_j1 = (castiguri_j1 / numar_jocuri) * 100

    return procentaj_j0, procentaj_j1

# Simulam 10.000 de jocuri si afisam rezultatele
rezultate = simulare_multipla(10000)
print(f"Procentaj de castig pentru j0: {rezultate[0]:.2f}%")
print(f"Procentaj de castig pentru j1: {rezultate[1]:.2f}%")

#punctul 2 si 3
# Definim reteaua Bayesiana
model = BayesianModel([('CineIncepe', 'StemeJ0'), ('CineIncepe', 'StemeJ1'), ('StemeJ0', 'Castigator')])

# Calculam parametrii modelului folosind date simulate
data = []
for _ in range(10000):
    cine_incepe = random.choice([0, 1])
    steme_j0 = arunca_moneda_normala()
    steme_j1 = arunca_moneda_masluita()

    if cine_incepe == 0:
        castigator = 0 if steme_j0 >= steme_j1 else 1
    else:
        castigator = 1 if steme_j1 > steme_j0 else 0

    data.append({'CineIncepe': cine_incepe, 'StemeJ0': steme_j0, 'StemeJ1': steme_j1, 'Castigator': castigator})

# Estimam parametrii modelului
model.fit(data, estimator=ParameterEstimator)

# Determinam cine a inceput jocul folosind inferenta bayesiana
inference = VariableElimination(model)
print(inference.query(variables=['CineIncepe'], evidence={'StemeJ1': 1}))


#Subiectul2

#punctul 1
# Generam 1000 de timp medii de asteptare folosind distributia beta
alpha = 2
beta = 5
timp_mediu_asteptare = np.random.beta(alpha, beta, 1000)

# Vizualizam distributia rezultata
plt.hist(timp_mediu_asteptare, bins=30, density=True, alpha=0.5, color='b')
plt.title('Distribuție a timpilor medii de așteptare')
plt.xlabel('Timp mediu de așteptare')
plt.ylabel('Densitate de probabilitate')
plt.show()



#punctul 2 si 3
# Date observate - timpul de așteptare la coadă
timp_asteptare_observat = [10, 15, 20, 25, 30]

# Definim modelul în PyMC
with pm.Model() as model_asteptare:
    # Definim prior pentru parametrul alfa al distributiei Gamma
    alfa = pm.Uniform('alfa', lower=0, upper=10)
    # pm.Uniform este folosit pentru a defini o distribuție uniforma pentru alegerea valorilor initiale ale parametrilor
    # Definim prior pentru parametrul beta al distributiei Gamma
    beta = pm.Uniform('beta', lower=0, upper=10)

    # Definim distributia Gamma pentru timpul de așteptare
    timp_asteptare = pm.Gamma('timp_asteptare', alpha=alfa, beta=beta, observed=timp_asteptare_observat)

    # Aplicam metoda MCMC pentru a estima distribuția a posteriori
    trace = pm.sample(2000, tune=1000, cores=1)

# Vizualizam distributia a posteriori pentru parametrul alfa
az.plot_posterior(trace, var_names=['alfa'], kind='hist', bins=30, hdi_prob=0.95)
plt.title('Distributia a posteriori pentru alfa')
plt.show()
#La subiectul 2 , pentru a compila , trebuie comentat subiectul 1 si invers