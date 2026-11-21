from package.PINNS_creation import Pinns
from package.PINNS_fourrier_embedding_creation import Pinns_Fourier
import functools
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from torch import nn

#IF THE COMPUTER HAS A GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#PATH FOR PLOTS
dossier = Path("outputs/graphs")

#HYPERPARAMETERS
n_hidden_layers = 3
loss_regularisation_wheight = 0.05
loss_differential_equation_wheight = 3
loss_boundary_weight = 1
n_nodes_per_layer = 126

#TRAINING PARAMETERS
epochs = 500
lr = 5e-3

#LOSS FUNCTIONS

#on créer la régularisation, on fait la somme des carré des paramètres du modèle
def loss_regularisation_function(model: nn.Module):
    reg = torch.zeros((), device=next(model.parameters()).device)
    for name, p in model.named_parameters():
        if "weight" in name:                      # on régularise les poids, pas les biais
            reg = reg + torch.sum(p.pow(2))
    return reg


def grad(outputs, inputs):
    """Computes the partial derivative of 
    an output with respect to an input.
    Args:
        outputs: (N, 1) tensor
        inputs: (N, D) tensor
    """
    return torch.autograd.grad(
        outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True
    ) 

def physics_loss_function_cooling_1D(model: torch.nn.Module,time): #on défini la loss physique liée à léquation diff
    Tenv = 25
    R = 0.005
    T = model(time)
    dT_dt = grad(T, time)[0] #on calcule le gradient de temperature par rapport au temps, le dT/dt dans l'équation
    pde =  R*(Tenv - T) - dT_dt#l'équation diff, avec tout les termes passé d'un côté donc si la solution est bonne, doit être égal à 0
    
    return torch.mean(pde**2) 


loss_residual = physics_loss_function_cooling_1D
loss_regularisation = loss_regularisation_function 

#NETWORK CREATION
pinn = Pinns(input_dim=1,output_dim=1,
                          n_hidden_layers=n_hidden_layers,
                          loss_residual_DE=loss_residual,
                          loss_regularisation= loss_regularisation,
                          loss_regularisation_weight=loss_regularisation_wheight,
                          loss_residual_DE_weight=loss_differential_equation_wheight,
                          loss_boundary_weight= loss_boundary_weight,
                          n_nodes_per_layer=n_nodes_per_layer,
                          epochs=epochs,
                          lr= lr
                          )

# #NETWORK CREATION
# pinn = Pinns_Fourier(input_dim=1,output_dim=1,
#                           n_hidden_layers=n_hidden_layers,
#                           loss_residual_DE=loss_residual,
#                           loss_regularisation= loss_regularisation,
#                           loss_regularisation_weight=loss_regularisation_wheight,
#                           loss_residual_DE_weight=loss_differential_equation_wheight,
#                           loss_boundary_weight= loss_boundary_weight,
#                           n_nodes_per_layer=n_nodes_per_layer,
#                           epochs=epochs,
#                           lr= lr,
#                           sigma_fourier=0.0003,
#                           number_fourier_frequency=5
#                           )





#EQUATION SOLUTION 
def cooling_law(time, Tenv, T0, R):
    T = Tenv + (T0 - Tenv) * np.exp(-R * time)
    return T

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
n_collocation_points = 2000
time_collocation = torch.linspace(0, 1000, steps=n_collocation_points,).view(-1,1).requires_grad_(True).to(DEVICE)

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

plt.plot(times, temps, label = "Real cooling law")
plt.plot(times, t_pred,label = "Network prediction ")
plt.legend()
plt.ylabel('Tempereature (C°)')
plt.xlabel('Time (s)')
plt.savefig("OUTPUT")


