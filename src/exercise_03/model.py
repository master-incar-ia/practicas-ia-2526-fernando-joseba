# Define los modelos de perceptrón simple y multicapa/multilayer perceptron

import torch
import torch.nn as nn

# Debe tener el init y el forward
class SimplePerceptron(nn.Module): # Hereda de nn.Module, es un requisito
    """
    Modelo de perceptrón simple: una sola capa lineal y una activación identidad
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = nn.Identity()

    def forward(self, x, use_activation=True):
        x = self.fc(x)
        if use_activation:
            x = self.activation(x)
        return x


class MultiLayerPerceptron(nn.Module):  # Hereda de nn.Module, es un requisito
    """ 
    Modelo de perceptrón multicapa: Capa oculta con ReLU y capa de salida lineal
    """
    def __init__(self, input_dim, output_dim, num_hidden_neurons, apodo): # Parametros de entrada y salida
        super().__init__() # es necesario
        self.fc1 = nn.Linear(input_dim, num_hidden_neurons) # Modelo lineal , añade capa lineal del perceptron, "capa oculta"
        self.fc2 = nn.Linear(num_hidden_neurons, output_dim) # Capa de salida
        self.activation = nn.ReLU()
        self.apodo = apodo

    def forward(self, x, use_activation=True):
        x1 = self.fc1(x)  # Falta la activacion
        x1 = self.activation(x1)  # Aplicamos la activacion con ReLU
        x2 = self.fc2(x1) # Capa de salida

        if use_activation:
            x2 = self.activation(x2)
        return x2


if __name__ == "__main__": # Para probar si el modelo funciona
    model = MultiLayerPerceptron(1, 1, 64, "exercise_03")
    print(model)
    x = torch.tensor([1.0])
    print(model(x))
    pass