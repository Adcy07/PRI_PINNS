from package.PINNS_creation import Pinns 
from package.physical_problem import physics_loss_function,loss_regularisation_function,analitycal_solution
import functools
import matplotlib.pyplot as plt
import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pinn = Pinns(input_dim=1,output_dim=1,
                          n_hidden_layers=3,
                          loss_differential_equation=physics_loss_function,
                          loss_regularisation= loss_regularisation_function,
                          loss_regularisation_wheight=0.05,
                          loss_differential_equation_wheight= 1
                          )



times = np.linspace(-2000, 0.5, 1000)
eq = functools.partial(analitycal_solution)
temps = eq(times)



t_init = np.array([0])
temp_init = np.array([1])

boundary_conditions = t_init,temp_init


plt.figure()
losses = pinn.fit(boundary_conditions)
plt.plot(losses)
plt.yscale('log')

t = np.linspace(-2000, 0.5, num = 300)

t_pred = pinn.predict(t)

plt.figure()
plt.plot(times, temps, label = "Equation")
plt.plot(t, t_pred,label = "Prédiction modèle ") #affichage de la solution du modèle 
plt.legend()
plt.ylabel('Temperature (C)')
plt.xlabel('Time (s)')

plt.show()
