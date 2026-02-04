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
    Modelo de perceptrón multicapa: tres capas lineales y activaciones ReLU
    """
    def __init__(
        self, input_dim, output_dim, num_hidden_neurons, apodo
    ):  # Parametros de entrada y salida
        super().__init__()  # es necesario
        self.fc1 = nn.Linear(
            input_dim, num_hidden_neurons
        )  # Modelo lineal , añade capa lineal del perceptron, "primera capa oculta"
        self.fc2 = nn.Linear(num_hidden_neurons, num_hidden_neurons)  # Segunda capa oculta
        self.fc3 = nn.Linear(num_hidden_neurons, output_dim)  # Capa de salida
        self.activation = nn.ReLU()
        self.apodo = apodo

    def forward(self, x, use_activation=True):
        x1 = self.fc1(x)  # Primera capa
        x1 = self.activation(x1)  # Aplicamos la activacion
        x2 = self.fc2(x1)  # Segunda capa oculta
        x2 = self.activation(x2)  # Aplicamos la activacion
        x3 = self.fc3(x2)  # Capa de salida

        if use_activation:
            x3 = self.activation(x3)
        return x3


if __name__ == "__main__":  # Para probar si el modelo funciona
    model = SimplePerceptron(1, 1)
    print(model)
    x = torch.tensor([1.0])
    print(model(x))
    pass
