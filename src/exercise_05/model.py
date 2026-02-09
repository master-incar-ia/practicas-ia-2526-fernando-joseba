# Define los modelos de perceptrón simple y multicapa/multilayer perceptron

import torch
import torch.nn as nn
import torch.nn.functional as F # Funciones de activación, etc.

# Debe tener el init y el forward
class SimplePerceptron(nn.Module): # Hereda de nn.Module, es un requisito
    """
    Modelo de perceptrón simple: una sola capa lineal y una activación identidad
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim) # Capa fully conected (lineal)
        self.activation = nn.Identity()

    def forward(self, x, use_activation=True):
        x = self.fc(x)
        if use_activation:
            x = self.activation(x)
        return x


class MultiLayerPerceptron(nn.Module): # Hereda de nn.Module, es un requisito
    """ 
    Modelo de perceptrón multicapa con 2 capas ocultas:
        input -> fc1 -> ReLU -> fc2 -> ReLU -> fc3 (salida)
    use_activation debe ser False para que la salida dé valores positivos y negativos
    """
    def __init__(self, input_dim, output_dim, num_hidden_neurons, apodo): # Parametros de entrada y salida
        super().__init__() # Es necesario
        self.fc1 = nn.Linear(input_dim, num_hidden_neurons) # Capa fully conected (lineal)
        self.fc2 = nn.Linear(num_hidden_neurons, num_hidden_neurons)
        self.fc3 = nn.Linear(num_hidden_neurons, output_dim)

        self.activation = nn.ReLU()
        self.apodo = apodo

    def forward(self, x, use_activation=False): # Se pone la última ReLU a False para que la salida pueda ser positiva o negativa, ya que la función original tiene valores positivos y negativos
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)

        if use_activation:
            x = self.activation(x)
        return x

class MultiLayerPerceptron_05(nn.Module): # Hereda de nn.Module, es un requisito
    """ 
    Modelo de perceptrón multicapa con 2 capas ocultas para exercise_05:
        input -> fc1 -> ReLU -> fc2 -> ReLU -> fc3 (salida)
    """
    def __init__(self, input_dim, output_dim, num_hidden_neurons): # Parametros de entrada y salida
        super().__init__() # Es necesario
        self.fc1 = nn.Linear(input_dim, num_hidden_neurons) # Capa fully conected (lineal)
        self.fc2 = nn.Linear(num_hidden_neurons, num_hidden_neurons)
        self.fc3 = nn.Linear(num_hidden_neurons, output_dim)

    def forward(self, x): # Se pone la última ReLU a False para que la salida pueda ser positiva o negativa, ya que la función original tiene valores positivos y negativos
        x = self.flatten(x) # Aplanar la imagen de 3D a 1D
        x = F.relu(self.fc1(x)) # Primera transformación lineal + activación ReLU
        x = F.relu(self.fc2(x)) # Segunda transformación lineal + activación ReLU
        x = self.fc3(x) # Capa de salida sin activación, ya que es un problema de clasificación y se usará CrossEntropyLoss que incluye softmax
        return x

if __name__ == "__main__": # Para probar si el modelo funciona
    model = SimplePerceptron(1, 1)
    print(model)
    x = torch.tensor([1.0])
    print(model(x))
    pass