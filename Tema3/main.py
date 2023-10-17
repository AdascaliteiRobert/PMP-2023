from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Definim structura rețelei bayesiene
model = BayesianNetwork([('Cutremur', 'Incendiu'), ('Cutremur', 'Alarmă'), ('Incendiu', 'Alarmă')])

# Definim Tabelele de probabilități condiționate (CPD)
cpd_cutremur = TabularCPD(variable='Cutremur', variable_card=2, values=[[0.9995], [0.0005]])
cpd_incendiu = TabularCPD(variable='Incendiu', variable_card=2, values=[[0.99, 0.97], [0.01, 0.03]], evidence=['Cutremur'], evidence_card=[2])
cpd_alarmă = TabularCPD(variable='Alarmă', variable_card=2, values=[[0.9999, 0.05, 0.98 , 0.02], [0.0001, 0.95, 0.02, 0.98]], evidence=['Incendiu', 'Cutremur'], evidence_card=[2, 2])

# Adăugăm CPD-urile la rețeaua bayesiană
model.add_cpds(cpd_cutremur, cpd_incendiu, cpd_alarmă)

# Verificăm dacă rețeaua bayesiană este validă
model.check_model()

# Creăm un obiect pentru inferență
inference = VariableElimination(model)

# Calculăm probabilitatea că a avut loc un cutremur, dat fiind că alarma de incendiu a fost declanșată
result = inference.query(variables=['Cutremur'], evidence={'Alarmă': 1})

print(result)

result = inference.query(variables=['Incendiu'] , evidence={'Alarmă':0})

print(result)