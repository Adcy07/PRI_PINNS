import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as thdat
import numpy as np


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Tenv = 25
T0 = 100
R = 0.005


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

def physics_loss_function(model: torch.nn.Module): #on défini la loss physique liée à léquation diff
    time = torch.linspace(0, 1000, steps=1000,).view(-1,1).requires_grad_(True).to(DEVICE) 
    temperatures = model(time)
    dT = grad(temperatures, time)[0] #on calcule le gradient de temperature par rapport au temps, le dT/dt dans l'équation
    pde = R*(Tenv - temperatures) - dT #l'équation diff, avec tout les termes passé d'un côté donc si la solution est bonne, doit être égal à 0
    
    return torch.mean(pde**2) 

# #on créer la régularisation, on fait la somme des carré des paramètres du modèle
# def loss_regularisation_function(model: torch.nn.Module):
#     return torch.sum(sum([p.pow(2.) for p in model.parameters()]))
