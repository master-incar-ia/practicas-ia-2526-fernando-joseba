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


class MultiLayerPerceptron(nn.Module): # Hereda de nn.Module, es un requisito
    """
    Modelo de perceptrón multicapa: Capa oculta con ReLU y capa de salida lineal
    """
    def __init__(
        self, input_dim, output_dim, num_hidden_neurons, apodo
    ):  # Parametros de entrada y salida
        super().__init__()
        self.fc1 = nn.Linear(input_dim, num_hidden_neurons)
        self.fc2 = nn.Linear(num_hidden_neurons, output_dim)
        self.activation = nn.Identity()
        self.activation = nn.ReLU()
        self.apodo = apodo

    def forward(self, x, use_activation=True):
        x1 = self.fc1(x)  # Falta la activación
        x1 = self.activation(x1)  # Aplicar la activación, relu porque es no lineal
        x2 = self.fc2(x1)

        if use_activation:
            x2 = self.activation(x2)
        return x2


class DeepMultiLayerPerceptron(nn.Module):
    """
    Modelo de perceptrón multicapa profundo: Dos capas ocultas con ReLU y capa de salida lineal
    """
    def __init__(self, input_dim, output_dim, hidden_dim, apodo):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.activation = nn.ReLU()
        self.apodo = apodo

    def forward(self, x, use_activation=True):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)

        # Nota: el "use_activation" controla si activamos la última capa o no.
        if use_activation:
            x = self.activation(x)

        return x


if __name__ == "__main__":
    model = SimplePerceptron(1, 1)
    print(model)
    x = torch.tensor([1.0])
    print(model(x))
    pass
