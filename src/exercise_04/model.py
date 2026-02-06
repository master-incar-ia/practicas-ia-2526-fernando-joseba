# Define los modelos de multicapa/multilayer perceptron

import torch
import torch.nn as nn

class MultiLayerPerceptron(nn.Module):  # Hereda de nn.Module, es un requisito
    """
    Vamos a crear una red convolucional para clasificar las imágenes de CIFAR-10. La arquitectura será la siguiente:
    """
    def __init__(
        self, in_channels, out_channels, kernel_size, padding):  # Parametros de entrada y salida
        super().__init__()  # es necesario
        self.fc1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding) # Capa convolucional (entrada, filtros, tamaño del filtro, reducir tamaño)
        self.fc2 = nn.ReLU(inplace=True)  # Activación ReLU
        self.fc3 = nn.Conv2d(in_channels, out_channels, kernel_size, padding) 
        self.fc4 = nn.ReLU(inplace=True)  # Capa de salida
        self.activation = nn.ReLU()
      

    def forward(self, x, use_activation=True):
        x1 = self.fc1(x)  # Primera capa oculta
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