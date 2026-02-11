# Script de evaluación y generación de gráficos

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

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


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Construye la matriz de confusión (num_classes x num_classes).
    Filas: clase real
    Columnas: clase predicha
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def plot_confusion_matrix(cm: np.ndarray, class_names, output_folder: Path) -> None:
    """
    Dibuja y guarda la matriz de confusión como imagen.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    # Escribir valores en cada celda
    thresh = cm.max() * 0.6
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, int(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(output_folder / "confusion_matrix.png")
    plt.close()


def evaluate(loader: DataLoader, model: nn.Module, criterion: nn.Module, device: torch.device):
    """
    Evalúa un dataloader:
    - loss medio
    - accuracy
    - vectores y_true / y_pred (para matriz de confusión)
    """
    model.eval()  # modo evaluación (desactiva dropout/bn si existieran)
    total_loss = 0.0
    correct = 0
    total = 0
    y_true_list = []
    y_pred_list = []

    with torch.no_grad():  # sin gradientes para ir más rápido y gastar menos memoria
        for inputs, targets in loader:
            # Mover a device (GPU/CPU)
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward: el modelo devuelve logits (B, 10)
            logits = model(inputs)

            # Loss de clasificación
            loss = criterion(logits, targets)
            total_loss += loss.item()

            # Predicción: clase con mayor logit
            preds = torch.argmax(logits, dim=1)

            # Accuracy
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            # Guardar para matriz de confusión (pasamos a CPU -> numpy)
            y_true_list.append(targets.detach().cpu().numpy())
            y_pred_list.append(preds.detach().cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = correct / total

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)

    return avg_loss, acc, y_true, y_pred


def evaluate_and_plot(loader, model, dataset_name, output_folder):
    model.eval()  # Poner el modelo en modo evaluación para poder desactivar dropout, batchnorm, etc.
    all_inputs = []  # X
    all_outputs = []  # Y'
    all_targets = []  # Y

    with torch.no_grad():  # Desactivar el cálculo de gradientes para ir más rápido
        for inputs, targets in loader:
            outputs = model(inputs)
            all_inputs.append(inputs.numpy())
            all_outputs.append(outputs.numpy())
            all_targets.append(targets.numpy())

    all_inputs = np.concatenate(all_inputs)
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)

    df = pd.DataFrame(
        data=np.array(
            [all_inputs.flatten(), all_targets.flatten(), all_outputs.flatten()]
        ).transpose(),
        columns=["x", "y_true", "y_pred"],
    )

    # Calculate cross entropy loss, accuracy and confusion matrix
    cross_entropy_loss = nn.CrossEntropyLoss()(torch.tensor(all_outputs), torch.tensor(all_targets)).item() # Pérdida de entropía cruzada
    accuracy = np.mean(np.argmax(all_outputs, axis=1) == all_targets) # Exactitud de la clasificación

    metrics = {
        "cross entropy loss": cross_entropy_loss,
        "accuracy": accuracy,
    }

    print(f"Evaluation metrics for {dataset_name} dataset:")
    print(metrics)

    ax = sns.regplot(df, x="y_true", y="y_pred", label=dataset_name)
    ax.set_title(f"Regression plot for {dataset_name} dataset")
    plt.legend()
    plt.savefig(f"{output_folder}/{dataset_name}_regression_plot.png")
    plt.show()
    plt.close()

    # Plot the data points
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="x", y="y_true", label="True")
    sns.scatterplot(data=df, x="x", y="y_pred", label="Predicted")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Data points for {dataset_name} dataset")
    plt.legend()
    plt.savefig(f"{output_folder}/{dataset_name}_data_points_plot.png")
    plt.show()
    plt.close()

    return metrics


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
    batch_size=10
    pin_memory = True if device.type == "cuda" else False
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    # Load the best model weights
    input_dim=3*32*32 # Las imagenes CIFAR-10 son de 32x32 píxeles con 3 canales (RGB)
    output_dim=10 # CIFAR-10 tiene 10 clases
    num_hidden_neurons=128 # Número de neuronas en las capas ocultas
    model = MultiLayerPerceptron_05(input_dim=input_dim, output_dim=output_dim, num_hidden_neurons=num_hidden_neurons).to(device)
    model.load_state_dict(torch.load(output_folder / "best_model.pth"))

    metrics = {}
    # Evaluate and plot for train, validation and test datasets
    metrics["train"] = evaluate_and_plot(train_loader, model, "train", output_folder)
    metrics["validation"] = evaluate_and_plot(val_loader, model, "validation", output_folder)
    metrics["test"] = evaluate_and_plot(test_loader, model, "test", output_folder)

    # save  metrics as csv
    pd.DataFrame(metrics).to_csv(output_folder / "metrics.csv")

    # Save the metrics as an image
    save_metrics_as_picture(metrics, output_folder / "metrics.png")

    print("Evaluation complete!")








    # Loss de clasificación (logits + targets)
    criterion = nn.CrossEntropyLoss()

    # Nombres de clase (CIFAR-10)
    class_names = dataset.data.classes  # ['airplane', 'automobile', ...]

    # Evaluar en train/val/test y guardar resultados
    results_lines = []

    for split_name, loader in [("train", train_loader), ("validation", val_loader), ("test", test_loader)]:
        loss, acc, y_true, y_pred = evaluate(loader, model, criterion, device)

        print(f"{split_name}: loss={loss:.4f}  acc={acc:.4f}")
        results_lines.append(f"{split_name}: loss={loss:.6f}  acc={acc:.6f}")

        # Matriz de confusión + plot
        cm = compute_confusion_matrix(y_true, y_pred, num_classes=output_dim)
        plot_confusion_matrix(cm, class_names, output_folder)

    print("Evaluation complete!")