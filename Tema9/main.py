import pymc as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Citirea datelor din fisierul CSV
df = pd.read_csv('Admission.csv')

# Verificare pentru a asigura ca datele au fost citite corect
print(df.head())

# Crearea dictionarului 'data'
data = {
    'GRE': np.array(df['GRE']),
    'GPA': np.array(df['GPA']),
    'Admission': np.array(df['Admission'])
}


# simulare esantion din distributia a posteriori
with pm.Model() as logistic_model:
    trace = pm.sample(1000, tune=1000)

# Ex1
mean_betas = np.mean(trace['beta0']), np.mean(trace['beta1']), np.mean(trace['beta2'])
decision_boundary = -mean_betas[0] / mean_betas[2] - (mean_betas[1] / mean_betas[2]) * data['GRE']

hdi = pm.stats.hpd(trace, hdi_prob=0.94)

# Ex 2.
plt.scatter(data['GRE'], data['GPA'], c=data['Admission'], cmap='viridis')
plt.plot(data['GRE'], decision_boundary, label='Decision Boundary', color='red')
plt.fill_between(data['GRE'], hdi[:, 1] / mean_betas[2] + hdi[:, 0] / mean_betas[2], alpha=0.3, color='orange', label='94% HDI')
plt.xlabel('GRE Score')
plt.ylabel('GPA')
plt.legend()
plt.show()

# Ex 3.
new_student_data = {'GRE': 550, 'GPA': 3.5}
new_pi = pm.math.sigmoid(mean_betas[0] + mean_betas[1] * new_student_data['GRE'] + mean_betas[2] * new_student_data['GPA'])
new_hdi = pm.stats.hpd(new_pi)

# Ex 4
new_student_data_low = {'GRE': 500, 'GPA': 3.2}
new_pi_low = pm.math.sigmoid(mean_betas[0] + mean_betas[1] * new_student_data_low['GRE'] + mean_betas[2] * new_student_data_low['GPA'])
new_hdi_low = pm.stats.hpd(new_pi_low)

print(f"Interval HDI pentru probabilitatea de admitere pentru studentul cu scor 550 și GPA 3.5: {new_hdi}")
print(f"Interval HDI pentru probabilitatea de admitere pentru studentul cu scor 500 și GPA 3.2: {new_hdi_low}")
