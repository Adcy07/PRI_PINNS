import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as thdat
import math

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def np_to_th(x): #pour convertir tableau numpy en tenseur torch
    n_samples = len(x)
    return torch.from_numpy(x).to(torch.float).to(DEVICE).reshape(n_samples, -1)


class Pinns_Fourier(nn.Module): #un réseau hérite toujours de nn.Module
    def __init__( #initialisation du réseau
        self,
        input_dim, #nombre de variables d'entrée
        output_dim,#nombre de variables de sortie 
        n_nodes_per_layer,
        n_hidden_layers, 
        activation_hidden_layer,
        loss_residual_DE,#loss de l'équation diff aux points de collocation, fonction qu'on doit donner à la création du réseau
        loss_residual_DE_weight, #importance apportée à la loss issue de l'équation différentielle  
        loss_regularisation, #loss de regularisation, fonction qu'on doit donner à la création du réseau
        loss_regularisation_weight, #importance apportée à la loss issue de la régularization
        epochs, #nombre de passage sur toutes les points de colocation
        lr,#taux d'apprentisage, paramètre de la descente de gradient 
        loss_boundary_weight, #weight for the boudary loss
        sigma_fourier, 
        number_fourier_frequency,
        loss_boundary #loss sur les conditions aux limites    
    ) -> None:
        
        super().__init__()

        self.epochs = epochs
        self.loss_boundary = loss_boundary
        self.loss_residual_DE = loss_residual_DE 
        self.loss_regularisation_weight = loss_regularisation_weight
        self.loss_residual_DE_weight = loss_residual_DE_weight
        self.loss_boundary_weight = loss_boundary_weight
        self.loss_regularisation = loss_regularisation 
        self.lr = lr 
        self.sigma_fourier = sigma_fourier
        self.number_fourier_frequency  = number_fourier_frequency

        new_dimension = 2 * input_dim * self.number_fourier_frequency
       
        layers = [nn.Linear(new_dimension, n_nodes_per_layer), activation_hidden_layer] #creation du premier layer

        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(n_nodes_per_layer, n_nodes_per_layer), activation_hidden_layer]
        self.layers = nn.Sequential(*layers) 

        self.out = nn.Linear(n_nodes_per_layer, output_dim) #sépare le layer de sortie, il aurait pu être dans le layers

        for layer in self.modules(): #Initialise les poids avec la méthode glorot
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=1)
                nn.init.zeros_(layer.bias)

        self.to(DEVICE)
  
    def forward(self, x):
        self.tensor_B = torch.normal(0, self.sigma_fourier, size=(1,self.number_fourier_frequency))
        proj = 2 * math.pi * x * self.tensor_B         # (N, m) broadcast par B
        fourier_features = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)  # (N, 2m)
        h = self.layers(fourier_features)
        out = self.out(h)

        return out #retourne juste une inférence

    def fit(self,bourdary_conditions,bourdary_conditions_first_derivative,collocation):



        optimiser = optim.Adam(self.parameters(), lr=self.lr) #on initialise une instance de l'optimiseur
        self.train() #met le model en "mode train", active des options importantes
        losses = []
        for ep in range(self.epochs): #on fait autant de passage que d'epoch
            optimiser.zero_grad() #on remet le gradient à 0 à chaque epoch (d'habittude c'est à chaque batch au sein d'une epoch mais la on fait pas de batch on a pas de data)
            loss = self.loss_boundary(self,bourdary_conditions,bourdary_conditions_first_derivative)*self.loss_boundary_weight
            loss += self.loss_residual_DE_weight * self.loss_residual_DE(self,collocation)  #on ajoute les contribution des loss equa diff et weight, définies en tant que fonction indépendantes
            loss +=  self.loss_regularisation_weight * self.loss_regularisation(self)

            loss.backward()  # backward (gradients), les gradients sont stocké dans les layers du modèle
            optimiser.step() #on fait un pas de descente de gradient avec l'optimiser 
            losses.append(loss.item()) #on ajoute la loss de cette epoch
            if ep % int(self.epochs / 10) == 0:
                print(f"Epoch {ep}/{self.epochs}, loss: {losses[-1]:.2f}") #tout les 10 epoch on donne des infos dans le terminal
        print("entrainement fini")
        return losses

    def predict(self, X):
        print("inférence en cours")
        self.eval() #met le model en mode évalutation, desacvite les trucs superflus

        out = self.forward(np_to_th(X))
        return out.detach().cpu().numpy() #renvoi les prédiciton du model, comme model(X) mais là renvoi que ce qu'on veut 

