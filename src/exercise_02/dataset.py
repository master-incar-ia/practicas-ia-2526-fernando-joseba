# Define el dataset de regresión con ruido cuadrático

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import Dataset


class NoisyRegressionDataset(Dataset):
    """
    Genera el dataset con ruido siguiendo la ecuación:
    y = -3*x^2 + 5*x + delta
    donde delta es ruido gaussiano con desviación estándar `noise_std`
    """
    
    def __init__(self, noise_std=20, size=100, seed=42):
        np.random.seed(seed)
        self.x = np.random.uniform(0, 100, size=(size,))
        self.delta = np.random.normal(0, noise_std, size=(size,))
        self.y = -3 * self.x * self.x + 5 * self.x + self.delta

        # Create a DataFrame for visualization
        df = pd.DataFrame(data=np.array([self.x, self.y]).transpose(), columns=["x", "y"])
        self.df = df

        # Reshape for PyTorch compatibility
        self.x = self.x.reshape((-1, 1))
        self.y = self.y.reshape((-1, 1))

    def plot(self, filepath):
        # Guardar la gráfica de los datos
        ax = sns.scatterplot(self.df, x="x", y="y")
        ax.set_title("Synthetic noisy data of y=-3*x^2+5*x")
        plt.savefig(filepath)
        plt.show()

    def __len__(self):
        # Para que sea compatible con PyTorch Dataset
        return len(self.x)

    def __getitem__(self, idx):
        # Para que sea compatible con PyTorch Dataset
        # return self.x[idx], self.y[idx] # Se puede hacer así, pero es mejor usar tensores de torch
        return torch.tensor(self.x[idx], dtype=torch.float32), torch.tensor(
            self.y[idx], dtype=torch.float32
        )


if __name__ == "__main__":
    output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name
    output_folder.mkdir(exist_ok=True, parents=True)

    dataset = NoisyRegressionDataset()
    print(f"Dataset length: {len(dataset)}")
    print(f"First item: {dataset[0]}")
    # save the plot
    dataset.plot(output_folder / "plot_dataset_example.png")
