from package.PINNS_creation import Pinns,np_to_th
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
n_hidden_layers = 4
loss_regularisation_wheight = 0.05
loss_differential_equation_wheight = 1
loss_boundary_weight = 100
n_nodes_per_layer = 200

#TRAINING PARAMETERS
epochs = 300
lr = 1e-4

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

def physics_loss_function(model: torch.nn.Module,x): #on défini la loss physique liée à léquation diff
    y = model(x)
    dy_dx = grad(y, x)[0] #on calcule le gradient de temperature par rapport au temps, le dT/dt dans l'équation
    dy2_dx2 = grad(dy_dx,x)[0]
    pde = dy2_dx2 + (dy_dx)*2 #l'équation diff, avec tout les termes passé d'un côté donc si la solution est bonne, doit être égal à 0
    
    return torch.mean(pde**2) 

def boundary_condition_loss_function(model: torch.nn.Module, boundary_conditions, boundary_conditions_first_derivative):
    # Conditions sur y
    x_bc, y_bc = boundary_conditions
    x_bc = torch.as_tensor(x_bc, dtype=torch.float32, device=DEVICE).reshape(-1, 1)
    y_bc = torch.as_tensor(y_bc, dtype=torch.float32, device=DEVICE).reshape(-1, 1)

    # Conditions sur y'
    x_d, dy_bc = boundary_conditions_first_derivative
    x_d = torch.as_tensor(x_d, dtype=torch.float32, device=DEVICE).reshape(-1, 1)
    dy_bc = torch.as_tensor(dy_bc, dtype=torch.float32, device=DEVICE).reshape(-1, 1)
    x_d = x_d.clone().requires_grad_(True)   # AVANT le forward

    loss_y = torch.mean((model(x_bc) - y_bc) ** 2)

    y_d = model(x_d)
    dy_dx = torch.autograd.grad(
        y_d, x_d,
        grad_outputs=torch.ones_like(y_d),
        create_graph=True                     # indispensable pour l'entraînement
    )[0]
    loss_dy = torch.mean((dy_dx - dy_bc) ** 2)

    return loss_y + loss_dy




loss_residual = physics_loss_function
loss_regularisation = loss_regularisation_function 
loss_boundary = boundary_condition_loss_function

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
                          lr= lr,
                          loss_boundary = loss_boundary
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
def y_solution_function(x, C1, C2):
    y = np.log(x+C1)+C2
    return y

# CONSTANTS 

C1 = 1
C2 = 2

x = np.linspace(0, 10, 1000)
eq = functools.partial(y_solution_function,C1 = C1, C2 = C2)
y_solution = eq(x)

#BOUNDARY CONDITIONS
x_init = np.array([1-C1,0])
y_init = np.array([C2,np.log(C1)+C2])
boundary_conditions = x_init,y_init

x_init_deriv = np.array([0])
y_init_deriv = np.array([1/C1])
boundary_conditions_derivatives = x_init_deriv,y_init_deriv

#COLLOCATION POINTS
n_collocation_points = 10000
x_collocation = torch.linspace(0, 10, steps=n_collocation_points,).view(-1,1).requires_grad_(True).to(DEVICE)

#TRAINING AND COMPUTING THE LOSS
losses = pinn.fit(boundary_conditions,
                  boundary_conditions_derivatives,
                  x_collocation) 

#INFERENCE 
y_pred = pinn.predict(x)

#FIGURE FOR LOSSSES
plt.figure()
plt.plot(losses)
plt.yscale('log')
plt.xlabel("Epochs")
plt.ylabel("Loss")
name = "loss_lr" + str(lr) 



#FIGURE FOR INFERENCE RESULT
plt.figure()

plt.title("y''+(y')^2")
plt.plot(x, y_solution, label = "Equation solution")
plt.plot(x, y_pred,label = "Network prediction ")
plt.legend()
plt.ylabel('y')
plt.xlabel('x')
plt.savefig("OUTPUT")


