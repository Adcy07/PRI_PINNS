from package.PINNS_creation import Pinns 
from package.physical_problem import physics_loss_function,loss_regularisation_function,analitycal_solution,cooling_law
import functools
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

#IF THE COMPUTER HAS A GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#PATH FOR PLOTS
dossier = Path("outputs/graphs")

#HYPERPARAMETERS
n_hidden_layers = 4
loss_regularisation_wheight = 0.1
loss_differential_equation_wheight = 1
n_nodes_per_layer = 126

#TRAINING PARAMETERS
epochs = 300
lr = 1e-3

#LOSS FUNCTIONS
loss_residual = physics_loss_function
loss_regularisation = loss_regularisation_function 

#NETWORK CREATION
pinn = Pinns(input_dim=1,output_dim=1,
                          n_hidden_layers=n_hidden_layers,
                          loss_residual_DE=loss_residual,
                          loss_regularisation= loss_regularisation,
                          loss_regularisation_wheight=loss_regularisation_wheight,
                          loss_residual_DE_wheight=loss_differential_equation_wheight,
                          n_nodes_per_layer=n_nodes_per_layer,
                          epochs=epochs,
                          lr= lr
                          )


# CONSTANTS
Tenv = 25
T0 = 100
R = 0.005
times = np.linspace(0, 1000, 1000)
eq = functools.partial(cooling_law, Tenv=Tenv, T0=T0, R=R)
temps = eq(times)

#BOUNDARY CONDITIONS
t_init = np.array([0])
temp_init = np.array([100])
boundary_conditions = t_init,temp_init

#COLLOCATION POINTS
time_collocation = torch.linspace(0, 1000, steps=10000,).view(-1,1).requires_grad_(True).to(DEVICE)

#TRAINING AND COMPUTING THE LOSS
losses = pinn.fit(boundary_conditions,time_collocation) 

#INFERENCE 
t_pred = pinn.predict(times)

#FIGURE FOR LOSSSES
plt.figure()
plt.plot(losses)
plt.yscale('log')
plt.xlabel("Epochs")
plt.ylabel("Loss")
name = "loss_lr" + str(lr) 


#FIGURE FOR INFERENCE RESULT
plt.figure()
plt.title(f"Hyperparameter :(nodes per hidden layer = {n_nodes_per_layer},number of hidden layers = {n_hidden_layers} ")
plt.plot(times, temps, label = "Real cooling law")
plt.plot(times, t_pred,label = "Network prediction ")
plt.legend()
plt.ylabel('Tempereature (C°)')
plt.xlabel('Time (s)')
