"""PyTorch model architecture and dataset wrapper for MLForge.

Implements MLForgeClassifier: a feed-forward neural network with dynamically
sized input layer, 128-neuron ReLU hidden layer, 0.3 Dropout, 64-neuron ReLU hidden layer,
and a 1-logit output layer for binary classification on CPU.
"""

from typing import Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    """PyTorch Dataset wrapping tabular feature matrices and binary targets."""

    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        else:
            self.y = None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


class MLForgeClassifier(nn.Module):
    """Feed-forward neural network for tabular binary classification.

    Architecture:
        Input Layer: Dynamically sized to input_dim
        Linear(input_dim, hidden_dim1) -> ReLU -> Dropout(dropout_rate)
        Linear(hidden_dim1, hidden_dim2) -> ReLU
        Linear(hidden_dim2, 1) -> Output logit
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim1: int = 128,
        hidden_dim2: int = 64,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.dropout_rate = dropout_rate

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning unnormalized logits."""
        return self.network(x)

    def predict_proba(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Generate positive class probabilities by applying sigmoid to logits."""
        self.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                tensor_x = torch.tensor(X, dtype=torch.float32)
            else:
                tensor_x = X.to(torch.float32)
            logits = self.forward(tensor_x)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        return probs
