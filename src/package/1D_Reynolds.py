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
loss_regularisation_weight = 0.05
loss_differential_equation_wheight = 10
loss_boundary_weight = 10
n_nodes_per_layer = 100
activation = nn.ReLU()

#TRAINING PARAMETERS
epochs = 500
lr = 1e-3

#LOSS FUNCTIONS

#on créer la régularisation, on fait la somme des carré des paramètres du modèle
def loss_regularisation_function(model: nn.Module):
    reg = torch.zeros((), device=next(model.parameters()).device)
    for name, p in model.named_parameters():
        if "weight" in name:                      # on régularise les poids, pas les biais
            reg = reg + torch.sum(p.pow(2))
    return reg


def H_function(model,X): #the fluid's height (adimensionalised)
    H0 = 0.1 
    H = H0 + (X**2)/2

    return H

def physics_loss_function_cooling_1D(model, X):
    P = model(X)
    H = H_function(model,X)
    dP_dX = torch.autograd.grad(P, X, grad_outputs=torch.ones_like(P), create_graph=True)[0]
    left_hand_side = torch.autograd.grad(dP_dX*H**3, X, grad_outputs=torch.ones_like(dP_dX*H**3), create_graph=True)[0]
    righ_hand_side  = torch.autograd.grad(H, X, grad_outputs=torch.ones_like(H), create_graph=True)[0]
    residual  = left_hand_side- righ_hand_side

    return torch.mean(residual**2)

def boundary_condition_loss_function(model: torch.nn.Module, boundary_conditions, boundary_conditions_first_derivative):
    # Conditions sur y
    x_bc, p_bc = boundary_conditions

    x_bc = torch.as_tensor(x_bc, dtype=torch.float32, device=DEVICE).reshape(-1, 1)
    p_bc = torch.as_tensor(p_bc, dtype=torch.float32, device=DEVICE).reshape(-1, 1)

    loss_p = torch.mean((model(x_bc) - p_bc) ** 2)
  
    # Conditions sur y'
    x_d, dp_bc = boundary_conditions_first_derivative

    if len(x_d)>0: #cette ligne se fait uniquement si on a une condition sur la dérivée seconde
        x_d = torch.as_tensor(x_d, dtype=torch.float32, device=DEVICE).reshape(-1, 1).requires_grad_(True)
        dp_bc = torch.as_tensor(dp_bc, dtype=torch.float32, device=DEVICE).reshape(-1, 1)
        p_d = model(x_d)
        dy_dx = torch.autograd.grad(p_d, x_d,
        grad_outputs=torch.ones_like(p_d),
        create_graph=True            )[0]
        loss_dp = torch.mean((dy_dx - dp_bc) ** 2)

        

        return loss_dp + loss_p


    
    return loss_p




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


#EQUATION SOLUTION 


# CONSTANTS 
Xa = 0
Xb = 10

#BOUNDARY CONDITIONS
X_bound = np.array([0,4]) #normalisation, ici inutile car en 0
P_bound = np.array([0,0]) #normalisation
boundary_conditions = X_bound,P_bound

X_bound_deriv = np.array([])#normalisation, ici inutile car en 0
P_bound_deriv = np.array([]) #normalisation
boundary_conditions_derivatives = X_bound_deriv,P_bound_deriv


#COLLOCATION POINTS
n_collocation_points = 1000
X_collocation = (Xb*torch.rand(n_collocation_points)).view(-1,1).requires_grad_(True).to(DEVICE)


#TRAINING AND COMPUTING THE LOSS
losses = pinn.fit(boundary_conditions,boundary_conditions_derivatives,X_collocation) 

#INFERENCE
N_point_inference = 100 
X = np.linspace(Xa,Xb,N_point_inference)
P_pred = pinn.predict(X)    # entrée normalisée
#FIGURE FOR LOSSSES
plt.figure()
plt.plot(losses)
plt.yscale('log')
plt.xlabel("Epochs")
plt.ylabel("Loss")
name = "loss_lr" + str(lr) 



#FIGURE FOR INFERENCE RESULT
plt.figure()

plt.plot(X, P_pred,label = "Network prediction ")
plt.legend()
plt.ylabel('Pressure')
plt.xlabel('X')
file_name = f"OUTPUT_Reynolds_{str(activation)}_{n_hidden_layers}hidden_{n_hidden_layers}n_nodes_per_layer_{loss_regularisation_weight}regu_{loss_differential_equation_wheight}residual_{loss_boundary_weight}boudary_{epochs}epoch_{lr}lr_{n_collocation_points}collocation.png"
plt.savefig(file_name)
