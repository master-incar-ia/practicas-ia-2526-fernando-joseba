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


class ConvolutionalNeuralNetwork(nn.Module): # Hereda de nn.Module, es un requisito
    """
    Modelo de red concolucional con 2 capas ocultas para exercise_05:
        Input (3x32x32) -> Conv(3→32) + ReLU -> MaxPool (↓ tamaño a 16x16) -> Conv(32→64) + ReLU -> MaxPool (↓ tamaño a 8x8) -> Flatten -> Linear + ReLU -> Dropout -> Linear
    """

    def __init__(self, output_dim, num_hidden_neurons): # Parametros de salida y número de neuronas en las capas ocultas
        super().__init__()

        # Bloque convolucional 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) # Conv2d(in_channels, out_channels, kernel_size, padding).
        self.pool1 = nn.MaxPool2d(2, 2) # MaxPool2d(kernel_size, stride). Reduce el tamaño de la imagen a la mitad (de 32x32 a 16x16) y se queda con la información más importante de cada bloque de 2x2 píxeles. Esto ayuda a reducir el número de parámetros y a evitar el overfitting.

        # Bloque convolucional 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Clasificador
        self.flatten = nn.Flatten() # Aplanar la imagen de 3D a 1D para que pueda ser procesada por las capas lineales
        self.fc1 = nn.Linear(64 * 8 * 8, num_hidden_neurons) # Capa fully conected (lineal)
        self.dropout = nn.Dropout(p=0.5) # Dropout para evitar el overfitting
        self.fc2 = nn.Linear(num_hidden_neurons, output_dim)

    def forward(self, x):

        # Bloque 1
        x = F.relu(self.conv1(x)) # Primera transformación lineal + activación ReLU
        x = self.pool1(x)

        # Bloque 2
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # Clasificador
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) # logits (sin softmax)

        return x


if __name__ == "__main__": # Para probar si el modelo funciona
    model = SimplePerceptron(1, 1)
    print(model)
    x = torch.tensor([1.0])
    print(model(x))
    pass