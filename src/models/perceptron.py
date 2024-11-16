import numpy as np


class Perceptron:
    """
    Implementation of the Perceptron learning algorithm for binary classification.

    Attributes:
        weights (np.ndarray): Weight vector including bias
        learning_rate (float): Learning rate for weight updates
        epochs (int): Maximum number of training epochs
    """

    def __init__(self, input_dim: int, learning_rate: float = 0.01, epochs: int = 1000):
        """
        Initialize the Perceptron classifier.

        Args:
            input_dim (int): Number of input features
            learning_rate (float): Learning rate for weight updates (default: 0.01)
            epochs (int): Maximum number of training epochs (default: 1000)
        """
        self.weights = np.zeros(input_dim + 1)  # +1 for bias
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Perceptron model on the given data.

        Args:
            X (np.ndarray): Training features of shape (n_samples, n_features)
            y (np.ndarray): Target values of shape (n_samples,)
        """
        X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])

        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X_with_bias, y):
                net_input = np.dot(xi, self.weights)
                prediction = np.where(net_input >= 0.0, 1, -1)
                update = self.learning_rate * (target - prediction)
                self.weights += update * xi
                errors += int(update != 0.0)
            if errors == 0:
                break

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Args:
            X (np.ndarray): Features of shape (n_samples, n_features)

        Returns:
            np.ndarray: Predicted class labels (-1 or 1)
        """
        X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        net_input = np.dot(X_with_bias, self.weights)
        return np.where(net_input >= 0.0, 1, -1)