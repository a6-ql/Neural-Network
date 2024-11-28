import logging

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
from typing import Tuple, List


class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.label_encoders = {}
        self.fitted = False

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def get_target_column(self, data: pd.DataFrame) -> str:
        """Identify the target column name from possible variations."""
        possible_names = ['bird_category', 'bird category', 'category', 'bird', 'class']
        for name in possible_names:
            if name in data.columns:
                return name
        return data.columns[-1]

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate the input data."""
        if data.empty:
            raise ValueError("Input DataFrame is empty")

        required_columns = ['body_mass', 'beak_length', 'beak_depth', 'fin_length']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        data = data.replace('NA', np.nan)
        missing_counts = data.isnull().sum()
        if missing_counts.any():
            self.logger.info(f"Missing values found:\n{missing_counts[missing_counts > 0]}")
        return data

    def process_data(self, data: pd.DataFrame, fit: bool = True) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Process the input data by handling missing values, encoding categorical variables,
        and scaling numerical features.

        Args:
            data: Input DataFrame
            fit: Whether to fit the transformers (True for training, False for testing)

        Returns:
            Tuple containing:
            - Processed DataFrame
            - List of feature names
            - List of class labels
        """
        data = data.copy()

        # Validate input data
        self._validate_data(data)

        # Handle missing values
        data = self._handle_missing_values(data)

        # Identify target column
        target_column = self.get_target_column(data)

        # Split features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Process numerical features
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            if fit:
                X[numeric_cols] = self.scaler.fit_transform(self.imputer.fit_transform(X[numeric_cols]))
            else:
                X[numeric_cols] = self.scaler.transform(self.imputer.transform(X[numeric_cols]))

        # Process categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].fillna('missing'))
            else:
                # Handle unseen categories in test data
                X[col] = X[col].fillna('missing')
                X[col] = X[col].map(lambda x: x if x in self.label_encoders[col].classes_ else 'missing')
                X[col] = self.label_encoders[col].transform(X[col])

        # Encode target variable
        if fit:
            self.target_encoder = LabelEncoder()
            y = self.target_encoder.fit_transform(y)
        else:
            y = self.target_encoder.transform(y)

        # Combine features and target back into a single DataFrame
        processed_data = X.copy()
        processed_data[target_column] = y

        # Get feature names and class labels
        feature_names = X.columns.tolist()
        class_labels = self.target_encoder.classes_.tolist()

        return processed_data, feature_names, class_labels


class NeuralNetwork:
    def __init__(self, n_features, hidden_layers, n_classes, learning_rate, activation='sigmoid', use_bias=True):
        self.n_features = n_features
        self.hidden_layers = hidden_layers
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.use_bias = use_bias
        self.activation = activation

        # Initialize weights and biases
        self.weights = []
        self.biases = []

        # Input to first hidden layer
        prev_layer_size = n_features
        for hidden_size in hidden_layers:
            self.weights.append(np.random.randn(prev_layer_size, hidden_size) * 0.01)
            if use_bias:
                self.biases.append(np.random.randn(hidden_size) * 0.01)
            else:
                self.biases.append(np.zeros(hidden_size))
            prev_layer_size = hidden_size

        # Last hidden layer to output
        self.weights.append(np.random.randn(prev_layer_size, n_classes) * 0.01)
        if use_bias:
            self.biases.append(np.random.randn(n_classes) * 0.01)
        else:
            self.biases.append(np.zeros(n_classes))

    def activation_function(self, x):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        else:  # tanh
            return np.tanh(x)

    def activation_derivative(self, x):
        if self.activation == 'sigmoid':
            fx = self.activation_function(x)
            return fx * (1 - fx)
        else:  # tanh
            return 1 - np.tanh(x) ** 2

    def forward_propagation(self, X):
        self.activations = [X]
        self.z_values = []

        current_input = X
        for i in range(len(self.weights)):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            current_input = self.activation_function(z)
            self.activations.append(current_input)

        return current_input

    def backward_propagation(self, X, y):
        m = X.shape[0]

        # Convert y to one-hot encoding
        y_onehot = np.zeros((m, self.n_classes))
        y_onehot[np.arange(m), y] = 1

        # Calculate output layer error
        delta = self.activations[-1] - y_onehot

        # Initialize gradients
        weight_gradients = []
        bias_gradients = []

        # Calculate gradients for each layer
        for i in range(len(self.weights) - 1, -1, -1):
            weight_gradients.insert(0, np.dot(self.activations[i].T, delta) / m)
            bias_gradients.insert(0, np.mean(delta, axis=0))

            if i > 0:  # Not input layer
                delta = np.dot(delta, self.weights[i].T) * self.activation_derivative(self.z_values[i - 1])

        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            if self.use_bias:
                self.biases[i] -= self.learning_rate * bias_gradients[i]

    def train(self, X, y, epochs):
        for _ in range(epochs):
            # Forward propagation
            self.forward_propagation(X)

            # Backward propagation
            self.backward_propagation(X, y)

    def predict(self, X):
        output = self.forward_propagation(X)
        return np.argmax(output, axis=1)


class NeuralNetworkGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Neural Network Classifier")
        self.window.geometry("800x600")

        # Initialize data processor and logger
        self.data_processor = DataProcessor()
        self.logger = logging.getLogger(__name__)

        # Create main frame
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Dataset selection
        dataset_frame = ttk.LabelFrame(main_frame, text="Dataset Selection", padding="5")
        dataset_frame.pack(fill=tk.X, padx=5, pady=5)

        self.dataset_label = ttk.Label(dataset_frame, text="No dataset selected")
        self.dataset_label.pack(side=tk.LEFT, padx=5)

        ttk.Button(dataset_frame, text="Browse", command=self.load_dataset).pack(side=tk.RIGHT, padx=5)

        # Network parameters
        params_frame = ttk.LabelFrame(main_frame, text="Network Parameters", padding="5")
        params_frame.pack(fill=tk.X, padx=5, pady=5)

        # Create input fields
        ttk.Label(params_frame, text="Number of hidden layers:").pack()
        self.hidden_layers_entry = ttk.Entry(params_frame)
        self.hidden_layers_entry.pack()

        ttk.Label(params_frame, text="Neurons in each hidden layer (comma-separated):").pack()
        self.neurons_entry = ttk.Entry(params_frame)
        self.neurons_entry.pack()

        ttk.Label(params_frame, text="Learning rate:").pack()
        self.learning_rate_entry = ttk.Entry(params_frame)
        self.learning_rate_entry.pack()

        ttk.Label(params_frame, text="Number of epochs:").pack()
        self.epochs_entry = ttk.Entry(params_frame)
        self.epochs_entry.pack()

        # Bias checkbox
        self.use_bias = tk.BooleanVar()
        ttk.Checkbutton(params_frame, text="Use bias", variable=self.use_bias).pack()

        # Activation function dropdown
        ttk.Label(params_frame, text="Activation function:").pack()
        self.activation = tk.StringVar()
        activation_dropdown = ttk.Combobox(params_frame, textvariable=self.activation)
        activation_dropdown['values'] = ('sigmoid', 'tanh')
        activation_dropdown.pack()

        # Train button
        ttk.Button(main_frame, text="Train and Test", command=self.train_and_test).pack(pady=10)

        # Results display
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="5")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.results_text = tk.Text(results_frame, height=10, width=50)
        self.results_text.pack(fill=tk.BOTH, expand=True)

    def load_dataset(self):
        """Load and process the dataset from a file."""
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if file_path:
            self.dataset_path = file_path
            self.dataset_label.config(text=f"Dataset: {file_path.split('/')[-1]}")

            try:
                # Load the data
                data = pd.read_csv(file_path)

                # Process the data with fit=True for training data
                self.processed_data, self.feature_names, self.class_labels = self.data_processor.process_data(data,
                                                                                                              fit=True)

                # Update results text
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, f"Dataset loaded successfully!\n")
                self.results_text.insert(tk.END, f"Number of features: {len(self.feature_names)}\n")
                self.results_text.insert(tk.END, f"Number of classes: {len(self.class_labels)}\n")
                self.results_text.insert(tk.END, f"Total samples: {len(self.processed_data)}\n")

                # Log missing values information
                missing_values = data.isnull().sum()
                if missing_values.any():
                    self.results_text.insert(tk.END, "\nMissing values detected:\n")
                    for col, count in missing_values[missing_values > 0].items():
                        self.results_text.insert(tk.END, f"{col}: {count} missing values\n")

            except Exception as e:
                self.logger.error(f"Error loading dataset: {str(e)}")
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, f"Error loading dataset: {str(e)}")

    def prepare_train_test_split(self, data):
        """Prepare training and testing datasets."""
        # Get target column name
        target_column = self.data_processor.get_target_column(data)

        # Split features and target
        X = data.drop(columns=[target_column]).values
        y = data[target_column].values

        X_train = []
        X_test = []
        y_train = []
        y_test = []

        # Stratified split
        for class_idx in np.unique(y):
            class_mask = y == class_idx
            class_X = X[class_mask]
            class_y = y[class_mask]

            # Ensure we have enough samples
            n_samples = len(class_y)
            n_train = min(30, int(n_samples * 0.7))
            n_test = min(20, n_samples - n_train)

            X_train.append(class_X[:n_train])
            X_test.append(class_X[n_train:n_train + n_test])
            y_train.append(class_y[:n_train])
            y_test.append(class_y[n_train:n_train + n_test])

        return (np.vstack(X_train), np.vstack(X_test),
                np.concatenate(y_train), np.concatenate(y_test))

    def train_and_test(self):
        """Train and test the neural network with the current dataset."""
        if not hasattr(self, 'processed_data'):
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Please load a dataset first!")
            return

        try:
            # Get input values
            n_hidden_layers = int(self.hidden_layers_entry.get())
            neurons_per_layer = [int(x.strip()) for x in self.neurons_entry.get().split(',')]
            learning_rate = float(self.learning_rate_entry.get())
            epochs = int(self.epochs_entry.get())

            # Prepare data
            X_train, X_test, y_train, y_test = self.prepare_train_test_split(self.processed_data)

            # Create and train neural network
            nn = NeuralNetwork(
                n_features=len(self.feature_names),
                hidden_layers=neurons_per_layer,
                n_classes=len(self.class_labels),
                learning_rate=learning_rate,
                activation=self.activation.get(),
                use_bias=self.use_bias.get()
            )

            nn.train(X_train, y_train, epochs)

            # Calculate training accuracy
            y_train_pred = nn.predict(X_train)
            train_accuracy = accuracy_score(y_train, y_train_pred)

            # Test the model
            y_test_pred = nn.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            # Calculate confusion matrix for test set
            conf_matrix = confusion_matrix(y_test, y_test_pred)

            # Display results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Training completed!\n\n")
            self.results_text.insert(tk.END, f"Training Accuracy: {train_accuracy:.2%}\n")
            self.results_text.insert(tk.END, f"Test Accuracy: {test_accuracy:.2%}\n\n")
            self.results_text.insert(tk.END, f"Confusion Matrix:\n{conf_matrix}\n\n")

            # Log the results
            self.logger.info(f"Training completed with train accuracy: {train_accuracy:.2%} and test accuracy: {test_accuracy:.2%}")

        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Error during training: {str(e)}")


    def run(self):
        self.window.mainloop()


# Create and run the GUI
if __name__ == "__main__":
    gui = NeuralNetworkGUI()
    gui.run()