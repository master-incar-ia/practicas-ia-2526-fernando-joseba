# Script de evaluación y generación de gráficos

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from dataset import CIFAR10Dataset
from model import ConvolutionalNeuralNetwork


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


def evaluate_and_plot(loader, model, dataset_name, output_folder, device, class_names):
    model.eval() # Poner el modelo en modo evaluación para poder desactivar dropout, batchnorm, etc.
    all_inputs = []  # X (inputs)
    all_outputs = [] # Y' (salida del modelo: logits)
    all_targets = [] # Y (targets reales: clase 0..9)

    total_loss = 0.0 # Para calcular el loss medio
    correct = 0 # Para contar aciertos
    total = 0 # Total de muestras

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad(): # Desactivar el cálculo de gradientes porque no es necesario para evaluación
        for inputs, targets in loader:
            # Mover datos al device (GPU/CPU)
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward: el modelo devuelve logits (B, 10)
            outputs = model(inputs)

            # Loss de clasificación (CrossEntropy)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # Convertir logits -> clase predicha (argmax por fila)
            preds = torch.argmax(outputs, dim=1)

            # Accuracy (aciertos / total)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            all_inputs.append(inputs.detach().cpu().numpy()) # X
            all_outputs.append(outputs.detach().cpu().numpy()) # Y' (logits)
            all_targets.append(targets.detach().cpu().numpy()) # Y (clases reales)

    # Concatenar batches
    all_inputs = np.concatenate(all_inputs)
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)

    # En clasificación, "y_pred" no es un valor continuo, es la clase predicha.
    y_pred = np.argmax(all_outputs, axis=1)

    # "x": aquí no es un escalar, así que guardamos un índice de muestra (0..N-1)
    df = pd.DataFrame(
        data=np.array([np.arange(len(all_targets)), all_targets, y_pred]).transpose(),
        columns=["x", "y_true", "y_pred"],
    )

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
    }

    print(f"Evaluation metrics for {dataset_name} dataset:")
    print(metrics)

    # Construir matriz de confusión
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(all_targets, y_pred): 
        cm[int(t), int(p)] += 1

    sns.heatmap(cm, annot=True, cmap="Blues",  fmt="d", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion matrix for {dataset_name} dataset")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{dataset_name}_confusion_matrix.png")
    plt.show()
    plt.close()

    return metrics


def save_metrics_as_picture(metrics, filepath):
    # Create a DataFrame
    df = pd.DataFrame(metrics)

    # Round the values to 3 decimal places
    df = df.round(3)

    # Plot the table and save as an image
    fig, ax = plt.subplots(figsize=(8, 2)) # set size frame
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(
        cellText=df.values, colLabels=df.columns, rowLabels=df.index, cellLoc="center", loc="center"
    )

    # Save the plot as an image
    plt.savefig(filepath)


if __name__ == "__main__":
    # Set the seed for reproducibility
    torch.manual_seed(42)

    # Create output folder based on file folder
    output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name
    output_folder.mkdir(exist_ok=True, parents=True)

    device = get_device("auto") # choices are "auto", "cpu", "cuda"
    print(f"Using device: {device}")

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
    batch_size=64
    pin_memory = True if device.type == "cuda" else False
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    # Load the best model weights
    input_dim=3*32*32 # Las imagenes CIFAR-10 son de 32x32 píxeles con 3 canales (RGB)
    output_dim=10 # CIFAR-10 tiene 10 clases
    num_hidden_neurons=64 # Número de neuronas en las capas ocultas
    model = ConvolutionalNeuralNetwork(output_dim=output_dim, num_hidden_neurons=num_hidden_neurons).to(device)
    model.load_state_dict(torch.load(output_folder / "best_model.pth", map_location=device, weights_only=True))

    class_names = dataset.data.classes
    metrics = {}
    metrics["train"] = evaluate_and_plot(train_loader, model, "train", output_folder, device, class_names)
    metrics["validation"] = evaluate_and_plot(val_loader, model, "validation", output_folder, device, class_names)
    metrics["test"] = evaluate_and_plot(test_loader, model, "test", output_folder, device, class_names)

    # save  metrics as csv
    pd.DataFrame(metrics).to_csv(output_folder / "metrics.csv")

    # Save the metrics as an image
    save_metrics_as_picture(metrics, output_folder / "metrics.png")

    print("Evaluation complete!")