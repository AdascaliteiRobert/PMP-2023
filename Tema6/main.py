import pymc as pm
import matplotlib.pyplot as plt

Y_values = [0, 5, 10]
theta_values = [0.2, 0.5]

fig, axs = plt.subplots(len(Y_values), len(theta_values), figsize=(10, 10))

for i, Y in enumerate(Y_values):
    for j, theta in enumerate(theta_values):
        with pm.Model() as model:
            # Distributia a priori
            n = pm.Poisson('n', mu=10)

            # Distributia bin
            y = pm.Binomial('y', n=n, p=theta, observed=Y)

            # Distributia a posteriori
            trace = pm.sample(2000, tune=1000, cores=1)

        pm.plot_posterior(trace, ax=axs[i, j])
        axs[i, j].set_title(f'Y = {Y}, θ = {theta}')

plt.tight_layout()
plt.show()
