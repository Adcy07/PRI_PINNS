import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as thdat


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def np_to_th(x): #pour convertir tableau numpy en tenseur torch
    n_samples = len(x)
    return torch.from_numpy(x).to(torch.float).to(DEVICE).reshape(n_samples, -1)


class Pinns(nn.Module): #un réseau hérite toujours de nn.Module
    def __init__( #initialisation du réseau
        self,
        input_dim, #nombre de variables d'entrée
        output_dim,#nombre de variables de sortie 
        n_hidden_layers, 
        loss_residual_DE,#loss de l'équation diff aux points de collocation, fonction qu'on doit donner à la création du réseau
        loss_residual_DE_weight, #importance apportée à la loss issue de l'équation différentielle  
        loss_regularisation, #loss de regularisation, fonction qu'on doit donner à la création du réseau
        loss_regularisation_weight, #importance apportée à la loss issue de la régularization 
        n_nodes_per_layer, #nombre de noeuds dans les hiddens layers 
        epochs, #nombre de passage sur toutes les points de colocation
        lr,#taux d'apprentisage, paramètre de la descente de gradient 
        loss_boundary_weight, #weight for the boudary loss
        number_of_frequency,#number of frequency we choose for the fourrier embedding
        loss_boundary = nn.MSELoss() #loss sur les conditions aux limites    
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
        self.n_units = n_nodes_per_layer

        layers = [nn.Linear(input_dim, self.n_units), nn.Tanh()] #creation du premier layer

        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(self.n_units, self.n_units), nn.Tanh()]
        self.layers = nn.Sequential(*layers) 

        self.out = nn.Linear(self.n_units, output_dim) #sépare le layer de sortie, il aurait pu être dans le layers

        self.to(DEVICE)


    def forward(self, x):
        h = self.layers(x)
        out = self.out(h)
        

        return out #retourne juste une inférence

    def fit(self,bourdary_conditions,collocation):

        coordinates, conditions = bourdary_conditions

        coordinates = np_to_th(coordinates)
        conditions = np_to_th(conditions)

        optimiser = optim.Adam(self.parameters(), lr=self.lr) #on initialise une instance de l'optimiseur
        self.train() #met le model en "mode train", active des options importantes
        losses = []
        for ep in range(self.epochs): #on fait autant de passage que d'epoch
            optimiser.zero_grad() #on remet le gradient à 0 à chaque epoch (d'habittude c'est à chaque batch au sein d'une epoch mais la on fait pas de batch on a pas de data)
            prediction_boudaries = self.forward(coordinates)
            loss = self.loss_boundary(prediction_boudaries,conditions)*self.loss_boundary_weight
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

