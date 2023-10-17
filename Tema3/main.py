from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Definim structura retelei bayesiene
model = BayesianNetwork([('Cutremur', 'Incendiu'), ('Cutremur', 'Alarma'), ('Incendiu', 'Alarma')])

# Definim Tabelele de probabilitati conditionate (CPD)
cpd_cutremur = TabularCPD(variable='Cutremur', variable_card=2, values=[[0.9995], [0.0005]])
cpd_incendiu = TabularCPD(variable='Incendiu', variable_card=2, values=[[0.99, 0.97], [0.01, 0.03]], evidence=['Cutremur'], evidence_card=[2])
cpd_alarma = TabularCPD(variable='Alarma', variable_card=2, values=[[0.9999, 0.05, 0.98 , 0.02], [0.0001, 0.95, 0.02, 0.98]], evidence=['Incendiu', 'Cutremur'], evidence_card=[2, 2])

# Adaugam CPD-urile la reteaua bayesiana
model.add_cpds(cpd_cutremur, cpd_incendiu, cpd_alarmă)

# Verificam daca reteaua bayesiana este valida
model.check_model()

# Cream un obiect pentru inferenta
inference = VariableElimination(model)

# Calculam probabilitatea ca a avut loc un cutremur, dat fiind ca alarma de incendiu a fost declansata
result = inference.query(variables=['Cutremur'], evidence={'Alarma': 1})

print(result)

result = inference.query(variables=['Incendiu'] , evidence={'Alarma':0})

print(result)