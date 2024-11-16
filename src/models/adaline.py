import numpy as np


class Adaline:
    """
    Implementation of the Adaptive Linear Neuron (ADALINE) algorithm.

    Attributes:
        learning_rate (float): Learning rate for weight updates
        epochs (int): Maximum number of training epochs
        mse_threshold (float): Mean squared error threshold for early stopping
        weights (np.ndarray): Weight vector including bias
    """

    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000,
                 mse_threshold: float = 0.001):
        """
        Initialize the ADALINE classifier.

        Args:
            learning_rate (float): Learning rate for weight updates (default: 0.01)
            epochs (int): Maximum number of training epochs (default: 1000)
            mse_threshold (float): MSE threshold for early stopping (default: 0.001)
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.mse_threshold = mse_threshold
        self.weights = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the ADALINE model on the given data.

        Args:
            X (np.ndarray): Training features of shape (n_samples, n_features)
            y (np.ndarray): Target values of shape (n_samples,)
        """
        X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        self.weights = np.zeros(X_with_bias.shape[1])

        for _ in range(self.epochs):
            net_input = np.dot(X_with_bias, self.weights)
            errors = y - net_input
            self.weights += self.learning_rate * X_with_bias.T.dot(errors)
            mse = (errors ** 2).mean()
            if mse < self.mse_threshold:
                break

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Args:
            X (np.ndarray): Features of shape (n_samples, n_features)

        Returns:
            np.ndarray: Predicted clas
        """
        X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        net_input = np.dot(X_with_bias, self.weights)
        return np.where(net_input >= 0.0, 1, -1)