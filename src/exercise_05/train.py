# Script de entrenamiento

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from dataset import CIFAR10Dataset
from model import MultiLayerPerceptron_05


def get_device(force: str = "auto") -> torch.device:
    """
    Devuelve el dispositivo según 'force':
    - 'cpu'  -> CPU
    - 'cuda' -> GPU
    - 'auto' -> GPU si está disponible, si no CPU
    """
    force = force.lower()
    if force == "cpu":
        return torch.device("cpu")
    if force == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(output_folder: Path, device: torch.device):
    # Data augmentation
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))])

    # Create an instance of the dataset
    dataset = CIFAR10Dataset("./data", train=True, transform=transform)

    # Split the dataset into train, validation, and test sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create DataLoaders for the datasets
    batch_size=10
    pin_memory = True if device.type == "cuda" else False
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    # Define the model, loss function, and optimizer
    input_dim=3*32*32 # Las imagenes CIFAR-10 son de 32x32 píxeles con 3 canales (RGB)
    output_dim=10 # CIFAR-10 tiene 10 clases
    num_hidden_neurons=128 # Número de neuronas en las capas ocultas
    lr=0.001 # Learning rate (tasa de aprendizaje)
    model = MultiLayerPerceptron_05(input_dim=input_dim, output_dim=output_dim, num_hidden_neurons=num_hidden_neurons).to(device)
    criterion = nn.CrossEntropyLoss() # Función de pérdida. Se usa CrossEntropyLoss porque es un problema de clasificación
    optimizer = optim.AdamW(model.parameters(), lr=lr) # AdamW es el algoritmo de optimización que se usará para actualizar los pesos del modelo durante el entrenamiento

    # Training loop with validation and saving best weights
    num_epochs = 300 # Aumentamos de 100 a 400 porque el loss seguía bajando
    best_val_loss = float("inf")
    best_model_path = output_folder / "best_model.pth"

    train_losses = []
    val_losses = []

    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader: # input:x; target:y
            # Forward pass
            inputs_cuda = inputs.to(device)
            targets_cuda = targets.to(device)
            outputs = model(inputs_cuda) # outputs: y' (y gorro)
            loss = criterion(outputs, targets_cuda) # Pérdida

            train_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad() # Poner a cero los gradientes antes de la backward
            loss.backward() # Calcular los gradientes
            optimizer.step() # Actualizar los pesos

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation step
        model.eval() # Poner el modelo en modo evaluación para poder desactivar dropout, batchnorm, etc.
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs_cuda = inputs.to(device)
                targets_cuda = targets.to(device)

                outputs = model(inputs_cuda)
                loss = criterion(outputs, targets_cuda)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Guardar el mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
            )

    print(f"Best validation loss: {best_val_loss:.4f}, Model saved to {best_model_path}")

    # Plotting the training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs), train_losses, label="Train Loss")
    plt.plot(range(num_epochs), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")

    # Save the plot to the outs/ folder
    plt.savefig(output_folder / "loss_plot.png")

if __name__ == "__main__":
    # Set the seed for reproducibility
    torch.manual_seed(42)

    # Create output folder based on file folder
    output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name  
    output_folder.mkdir(exist_ok=True, parents=True)

    device = get_device("auto") # choices are "auto", "cpu", "cuda"
    print(f"Using device: {device}")
    train_model(output_folder, device=device)

    # Con 64 neuronas y 100 epocas: Best validation loss: 15594
    # Con 64 neuronas y 200 epocas: Best validation loss: 15363
    # Con 64 neuronas y 300 epocas: Best validation loss: 15269
    # Con 128 neuronas y 100 epocas: Best validation loss: 15061
    # Con 128 neuronas y 200 epocas: Best validation loss: 15015
    # Con 128 neuronas y 300 epocas: Best validation loss: 15015
    # Con 256 neuronas y 100 epocas: Best validation loss: 15090