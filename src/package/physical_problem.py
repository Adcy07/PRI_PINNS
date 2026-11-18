import torch
import torch.nn as nn
import numpy as np



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    time = torch.linspace(-1000, 0.5, steps=10000,).view(-1,1).requires_grad_(True).to(DEVICE) #création des points de collocation
    y = model(time)
    dy_dt = grad(y, time)[0] #on calcule le gradient de temperature par rapport au temps, le dT/dt dans l'équation
    pde =  dy_dt-y**2 #l'équation diff, avec tout les termes passé d'un côté donc si la solution est bonne, doit être égal à 0
    
    return torch.mean(pde**2) 

#on créer la régularisation, on fait la somme des carré des paramètres du modèle
def loss_regularisation_function(model: nn.Module):
    reg = torch.zeros((), device=next(model.parameters()).device)
    for name, p in model.named_parameters():
        if "weight" in name:                      # on régularise les poids, pas les biais
            reg = reg + torch.sum(p.pow(2))
    return reg

def analitycal_solution(time): #solution de notre équation diff, pour comparaison avec la solution du modèle
    y = 1/(1-time)
    return y