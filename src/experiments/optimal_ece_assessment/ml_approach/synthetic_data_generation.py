import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
This code is owned by Mohammad Hossein Shaker Ardakani, 
the tutor for the bachelor thesis this codebase belongs to.
"""


class SyntheticDataGenerator:
    def __init__(self, num_features=10, num_classes=2, hidden_layers=[32, 16], seed=42):
        """
        Initialize the synthetic data generator.

        Args:
        num_features (int): Number of input features (X).
        num_classes (int): Number of output classes (K). Use 2 for binary classification.
        hidden_layers (list): List specifying the number of neurons in each hidden layer.
        seed (int): Random seed for reproducibility.
        """
        self.num_features = num_features
        self.num_classes = num_classes
        self.seed = seed

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Define the neural network model
        layers = []
        input_dim = num_features

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(input_dim, num_classes))

        self.model = nn.Sequential(*layers)

        # Initialize weights randomly
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)

    def generate_data(self, num_samples=1000, temperature=1.0, mask_ratio=0.0):
        """
        Generate synthetic data.

        Args:
        num_samples (int): Number of samples to generate.
        temperature (float): Softmax temperature to control classification difficulty.
        mask_ratio (float): Ratio of features to mask (turn into noise).

        Returns:
        X (numpy.ndarray): Generated feature matrix (num_samples, num_features).
        Y (numpy.ndarray): Generated labels (num_samples,).
        """
        # Generate random feature matrix X
        X = np.random.randn(num_samples, self.num_features).astype(np.float32)

        # Mask certain features (turn into noise)
        if mask_ratio > 0:
            num_masked = int(self.num_features * mask_ratio)
            mask_indices = np.random.choice(self.num_features, num_masked, replace=False)
            X[:, mask_indices] = np.random.permutation(X[:, mask_indices])  # Shuffle to add noise

        X_tensor = torch.tensor(X)

        # Forward pass through the neural network to get logits
        logits = self.model(X_tensor)

        # Normalize outputs for each class between 0 and 1 before softmax
        logits_min = logits.min(dim=0, keepdim=True).values
        logits_max = logits.max(dim=0, keepdim=True).values
        normalized_logits = (logits - logits_min) / (logits_max - logits_min + 1e-8)  # Adding epsilon for numerical stability

        # Apply softmax with temperature scaling
        P = F.softmax(normalized_logits / temperature, dim=1).detach().numpy()

        # Sample Y from categorical distribution
        if self.num_classes == 2:
            Y = np.random.binomial(1, P[:, 1])  # Take the probability of class 1
        else:
            Y = np.array([np.random.choice(self.num_classes, p=p_row) for p_row in P])

        return X, Y, P