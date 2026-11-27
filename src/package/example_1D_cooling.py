from pinns_creation.PINNS_creation import Pinns
from pinns_creation.PINNS_fourrier_embedding_creation import Pinns_Fourier
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
loss_regularisation_weight = 0.1
loss_differential_equation_wheight = 1
loss_boundary_weight = 10
n_nodes_per_layer = 100
activation = nn.ReLU()

#TRAINING PARAMETERS
epochs = 2000
lr = 1e-3

#LOSS FUNCTIONS

#on créer la régularisation, on fait la somme des carré des paramètres du modèle
def loss_regularisation_function(model: nn.Module):
    reg = torch.zeros((), device=next(model.parameters()).device)
    for name, p in model.named_parameters():
        if "weight" in name:                      # on régularise les poids, pas les biais
            reg = reg + torch.sum(p.pow(2))
    return reg




def physics_loss_function_cooling_1D(model, time):
    Tmax = 1000
    R = 0.005
    k = R * Tmax
    theta = model(time)
    dtheta_dtau = torch.autograd.grad(theta, time, grad_outputs=torch.ones_like(theta), create_graph=True)[0]
    pde = -k * theta - dtheta_dtau   #
    return torch.mean(pde**2)

def boundary_condition_loss_function(model: torch.nn.Module, boundary_conditions, boundary_conditions_first_derivative):
    # Conditions sur y
    x_bc, y_bc = boundary_conditions

    x_bc = torch.as_tensor(x_bc, dtype=torch.float32, device=DEVICE).reshape(-1, 1)
    y_bc = torch.as_tensor(y_bc, dtype=torch.float32, device=DEVICE).reshape(-1, 1)

    loss_y = torch.mean((model(x_bc) - y_bc) ** 2)
  
    # Conditions sur y'
    x_d, dy_bc = boundary_conditions_first_derivative

    if len(x_d)>0: #cette ligne se fait uniquement si on a une condition sur la dérivée seconde
        x_d = torch.as_tensor(x_d, dtype=torch.float32, device=DEVICE).reshape(-1, 1).requires_grad_(True)
        dy_bc = torch.as_tensor(dy_bc, dtype=torch.float32, device=DEVICE).reshape(-1, 1)
        y_d = model(x_d)
        dy_dx = torch.autograd.grad(y_d, x_d,
        grad_outputs=torch.ones_like(y_d),
        create_graph=True            )[0]
        loss_dy = torch.mean((dy_dx - dy_bc) ** 2)

        return loss_dy + loss_y

    return loss_y




loss_residual = physics_loss_function_cooling_1D
loss_regularisation = loss_regularisation_function 
loss_boundary = boundary_condition_loss_function

#NETWORK CREATION
#NETWORK CREATION
pinn = Pinns(input_dim=1,output_dim=1,
                          n_hidden_layers=n_hidden_layers,
                          loss_residual_DE=loss_residual,
                          loss_regularisation= loss_regularisation,
                          loss_regularisation_weight=loss_regularisation_weight,
                          loss_residual_DE_weight=loss_differential_equation_wheight,
                          loss_boundary_weight= loss_boundary_weight,
                          n_nodes_per_layer=n_nodes_per_layer,
                          epochs=epochs,
                          lr= lr,
                          loss_boundary = loss_boundary,
                          activation_hidden_layer=activation
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
Tmax = 1000
times = np.linspace(0, Tmax, 1000)
eq = functools.partial(cooling_law, Tenv=Tenv, T0=T0, R=R)
temps = eq(times)

#BOUNDARY CONDITIONS
t_init = np.array([0/Tmax]) #normalisation, ici inutile car en 0
temp_init = np.array([100/T0]) #normalisation
boundary_conditions = t_init,temp_init

t_init_deriv = np.array([[0/Tmax]])#normalisation, ici inutile car en 0
T_init_deriv = np.array([-R * Tmax]) #normalisation
boundary_conditions_derivatives = t_init_deriv,T_init_deriv


#COLLOCATION POINTS
n_collocation_points = 1000
time_collocation = torch.rand(n_collocation_points).view(-1,1).requires_grad_(True).to(DEVICE)


#TRAINING AND COMPUTING THE LOSS
losses = pinn.fit(boundary_conditions,boundary_conditions_derivatives,time_collocation) 

#INFERENCE 
theta_pred = pinn.predict(times / Tmax)    # entrée normalisée
t_pred = Tenv + (T0 - Tenv) * theta_pred    # retour en °C
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
file_name = f"OUTPUT_Reynolds_{str(activation)}_{n_hidden_layers}hidden_{n_hidden_layers}n_nodes_per_layer_{loss_regularisation_weight}regu_{loss_differential_equation_wheight}residual_{loss_boundary_weight}boudary_{epochs}epoch_{lr}lr_{n_collocation_points}collocation.png"
plt.savefig(file_name)
