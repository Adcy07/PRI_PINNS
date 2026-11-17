import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as thdat
from package.physical_problem import physics_loss_function

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def np_to_th(x): #pour convertir tableau numpy en tenseur torch
    n_samples = len(x)
    return torch.from_numpy(x).to(torch.float).to(DEVICE).reshape(n_samples, -1)


class Pinns_without_data(nn.Module): #un réseau hérite toujours de nn.Module
    def __init__( #initialisation du réseau
        self,
        input_dim, #nombre de variables d'entrée
        output_dim,#nombre de variables de sortie 
        n_hidden_layers, 
        loss_differential_equation,#loss de l'équation diff aux points de collocation, fonction qu'on doit donner à la création du réseau
        loss_differential_equation_wheight, #importance apportée à la loss issue de l'équation différentielle  
        loss_regularisation, #loss de regularisation, fonction qu'on doit donner à la création du réseau
        loss_regularisation_wheight, #importance apportée à la loss issue de la régularization 
        n_units=100, #nombre de noeuds dans les hiddens layers 
        epochs=1000, #nombre de passage sur toutes les points de colocation
        lr=1e-3,#taux d'apprentisage, paramètre de la descente de gradient 
       
    ) -> None:
        super().__init__()

        self.epochs = epochs
        self.loss_function_differential_equation = loss_differential_equation 
        self.loss_regularisation_wheight = loss_regularisation_wheight
        self.loss_differential_equation_wheight = loss_differential_equation_wheight
        self.loss_regularisation = loss_regularisation 
        self.lr = lr 
        self.n_units = n_units

        layers = [nn.Linear(input_dim, self.n_units), nn.ReLU()] #creation du premier layer

        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(self.n_units, self.n_units), nn.ReLU()]
        self.layers = nn.Sequential(*layers) 

        self.out = nn.Linear(self.n_units, output_dim) #sépare le layer de sortie, il aurait pu être dans le layers

    def forward(self, x):
        h = self.layers(x)
        out = self.out(h)

        return out #retourne juste une inférence

    def fit(self, X, y):

        inputs = np_to_th(X) #tableau numpy vers tenseur qui continent les inputs

        optimiser = optim.Adam(self.parameters(), lr=self.lr) #on initialise une instance de l'optimiseur
        self.train() #met le model en "mode train", active des options importantes
        losses = [] #liste qui contiendra le loss à chaque epoch
        loss = 0
        for ep in range(self.epochs): #on fait autant de passage que d'epoch
            optimiser.zero_grad() #on remet le gradient à 0 à chaque epoch (d'habittude c'est à chaque batch au sein d'une epoch mais la on fait pas de batch on a pas de data)
            loss += self.loss_differential_equation_wheight * self.loss_function_differential_equation(self) + self.loss_regularisation_wheight*self.loss_regularisation #on ajoute les contribution des loss equa diff et weight, définies en tant que fonction indépendantes
            loss.backward()  # backward (gradients), les gradients sont stocké dans les layers du modèle, moyenné sur cette epoch automatiquement  
            optimiser.step() #on fait un pas de descente de gradient avec l'optimiser 
            losses.append(loss.item()) #on ajoute la loss de cette epoch
            if ep % int(self.epochs / 10) == 0:
                print(f"Epoch {ep}/{self.epochs}, loss: {losses[-1]:.2f}") #tout les 10 epoch on donne des infos dans le terminal
        return losses

    def predict(self, X):
        self.eval() #met le model en mode évalutation, desacvite les trucs superflus
        out = self.forward(np_to_th(X))
        return out.detach().cpu().numpy() #renvoi les prédiciton du model, comme model(X) mais là renvoi que ce qu'on veut 

