import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from typing import Optional, Dict, Any, Tuple

from ..models.perceptron import Perceptron
from ..models.adaline import Adaline
from ..utils.data_processor import DataProcessor


class NeuralNetworkApp:
    """
    A GUI application for performing neural network classification using the Perceptron and Adaline algorithms.
    Provides a user interface for loading datasets, selecting features and classes, setting training parameters,
    training a neural network model, and visualizing the model's decision boundaries and performance metrics.
    """

    def __init__(self, root: tk.Tk):
        """
        Initializes the main GUI application window and its components.

        Parameters:
            root (tk.Tk): The main application window object.
        """
        self.root = root
        self.root.title("Neural Network Classifier")
        self.root.geometry("650x500")

        # Initialize data processing utility, instance attributes, and GUI components
        self.data_processor = DataProcessor()
        self.initialize_attributes()
        self.create_widgets()

    def initialize_attributes(self) -> None:
        """
        Initializes instance attributes to hold data, features, class labels, and model-related parameters.
        This method resets attributes related to the dataset, feature selection, model training, and test data.
        """
        self.data = None                    # Placeholder for the loaded dataset
        self.features = []                   # List of available features in the dataset
        self.class_labels = []               # List of available class labels in the dataset
        self.model = None                    # Placeholder for the neural network model
        self.X_train = None                  # Training data features
        self.X_test = None                   # Test data features
        self.y_train = None                  # Training data labels
        self.y_test = None                   # Test data labels

    def create_widgets(self) -> None:
        """
        Creates and arranges the graphical user interface components.
        This includes buttons, labels, entry fields, combo boxes, and other elements that allow users
        to interact with the application.
        """
        # Main application frame
        frame = ttk.Frame(self.root, padding="10")
        frame.grid(row=0, column=0, sticky="NSEW", padx=10, pady=10)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Load Data Button
        self.load_button = ttk.Button(frame, text="Load Data", command=self.load_data)
        self.load_button.grid(row=0, column=0, columnspan=3, pady=10)

        # Feature Selection Drop-Downs
        ttk.Label(frame, text="Select Two Features:").grid(row=1, column=0, sticky="w", pady=10, padx=5)
        self.feature1_var = tk.StringVar()
        self.feature2_var = tk.StringVar()
        self.feature1_menu = ttk.Combobox(frame, textvariable=self.feature1_var, width=15, state="readonly")
        self.feature2_menu = ttk.Combobox(frame, textvariable=self.feature2_var, width=15, state="readonly")
        self.feature1_menu.grid(row=1, column=1, pady=10, padx=5)
        self.feature2_menu.grid(row=1, column=2, pady=10, padx=5)

        # Class Selection Drop-Downs
        ttk.Label(frame, text="Select Two Classes:").grid(row=2, column=0, sticky="w", pady=10, padx=5)
        self.class1_var = tk.StringVar()
        self.class2_var = tk.StringVar()
        self.class1_menu = ttk.Combobox(frame, textvariable=self.class1_var, width=15, state="readonly")
        self.class2_menu = ttk.Combobox(frame, textvariable=self.class2_var, width=15, state="readonly")
        self.class1_menu.grid(row=2, column=1, pady=10, padx=5)
        self.class2_menu.grid(row=2, column=2, pady=10, padx=5)

        # Training Parameters Input Fields
        self.create_parameter_inputs(frame)

        # Algorithm Selection and Training Controls
        self.create_training_controls(frame)

        # Bind events to update the feature and class selection menus dynamically
        self.bind_update_events()

    def create_parameter_inputs(self, frame: ttk.Frame) -> None:
        """
        Creates input fields for configuring model training parameters.
        These include learning rate, number of epochs, MSE threshold, and an option to add bias.

        Parameters:
            frame (ttk.Frame): The frame in which the parameter inputs will be placed.
        """
        # Learning rate entry
        ttk.Label(frame, text="Learning Rate (eta):").grid(row=3, column=0, sticky="w", pady=10, padx=5)
        self.eta_var = tk.DoubleVar(value=0.01)
        ttk.Entry(frame, textvariable=self.eta_var, width=22).grid(row=3, column=1, pady=10, padx=5)

        # Number of epochs entry
        ttk.Label(frame, text="Number of Epochs:").grid(row=4, column=0, sticky="w", pady=10, padx=5)
        self.epochs_var = tk.IntVar(value=1000)
        ttk.Entry(frame, textvariable=self.epochs_var, width=22).grid(row=4, column=1, pady=10, padx=5)

        # MSE threshold entry
        ttk.Label(frame, text="MSE Threshold:").grid(row=5, column=0, sticky="w", pady=10, padx=5)
        self.mse_var = tk.DoubleVar(value=0.001)
        ttk.Entry(frame, textvariable=self.mse_var, width=22).grid(row=5, column=1, pady=10, padx=5)

        # Bias checkbox
        self.bias_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Add Bias", variable=self.bias_var).grid(row=6, column=0, columnspan=3, pady=10)

    def create_training_controls(self, frame: ttk.Frame) -> None:
        """
        Creates radio buttons for algorithm selection and buttons for model training and testing.

        Parameters:
            frame (ttk.Frame): The frame in which training control elements will be placed.
        """
        # Algorithm selection radio buttons
        self.algorithm_var = tk.StringVar(value="Perceptron")
        ttk.Label(frame, text="Select Algorithm:").grid(row=7, column=0, sticky="w", pady=10, padx=5)
        ttk.Radiobutton(frame, text="Perceptron", variable=self.algorithm_var, value="Perceptron").grid(row=7, column=1)
        ttk.Radiobutton(frame, text="Adaline", variable=self.algorithm_var, value="Adaline").grid(row=7, column=2)

        # Train and Test buttons
        self.train_button = ttk.Button(frame, text="Train", command=self.train_model)
        self.train_button.grid(row=8, column=0, pady=20, padx=5)
        self.test_button = ttk.Button(frame, text="Test & Plot", command=self.test_model)
        self.test_button.grid(row=8, column=2, pady=20, padx=5)

    def bind_update_events(self) -> None:
        """
        Binds events to dynamically update feature and class selection menus
        when the user makes a selection.
        """
        self.feature1_var.trace("w", self.update_feature_options)
        self.feature2_var.trace("w", self.update_feature_options)
        self.class1_var.trace("w", self.update_class_options)
        self.class2_var.trace("w", self.update_class_options)

    def load_data(self) -> None:
        """
        Opens a file dialog to allow the user to select a CSV file, then loads and processes the data.
        Updates the feature and class menus with the loaded data's available features and classes.
        """
        file_path = filedialog.askopenfilename()
        if not file_path:
            return  # User canceled file selection

        try:
            # Load the CSV file and process it to extract features and labels
            self.data = pd.read_csv(file_path)
            self.data, self.features, self.class_labels = self.data_processor.process_data(self.data)

            # Update the GUI elements with the newly loaded data
            self.update_ui_after_load()
            messagebox.showinfo("Success", "Data loaded and preprocessed successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")

    def update_ui_after_load(self) -> None:
        """
        Updates the feature and class selection menus based on the loaded dataset.
        Called after successfully loading and processing data.
        """
        self.feature1_menu["values"] = self.features
        self.feature2_menu["values"] = self.features
        self.class1_menu["values"] = self.class_labels
        self.class2_menu["values"] = self.class_labels

    def update_feature_options(self, *args) -> None:
        """
        Dynamically updates available options for feature selection menus based on current selections.
        Prevents selecting the same feature in both menus.
        """
        if not self.features:
            return

        selected_feature1 = self.feature1_var.get()
        self.feature2_menu["values"] = [f for f in self.features if f != selected_feature1]

    def update_class_options(self, *args) -> None:
        """
        Dynamically updates available options for class selection menus based on current selections.
        Prevents selecting the same class in both menus.
        """
        if not self.class_labels:
            return

        selected_class1 = self.class1_var.get()
        self.class2_menu["values"] = [c for c in self.class_labels if c != selected_class1]

    def train_model(self) -> None:
        """
        Initializes and trains the selected model (Perceptron or Adaline) based on user-selected parameters.
        Displays a message upon successful training completion.
        """
        try:
            self.validate_selections()
            X, y = self.prepare_training_data()

            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )

            # Initialize and train model
            self.initialize_model()
            self.model.fit(self.X_train, self.y_train)

            messagebox.showinfo("Success", "Model trained successfully!")
            self.plot_decision_boundary()
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")

    def validate_selections(self) -> None:
        """Validate user selections before training."""
        if not all([self.feature1_var.get(), self.feature2_var.get(),
                    self.class1_var.get(), self.class2_var.get()]):
            raise ValueError("Please select both features and classes.")

    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training."""
        selected_features = [self.feature1_var.get(), self.feature2_var.get()]
        selected_classes = [self.class1_var.get(), self.class2_var.get()]

        data_filtered = self.data[self.data.iloc[:, -1].isin(selected_classes)]
        if data_filtered.empty:
            raise ValueError("No data available for selected classes.")

        X = data_filtered[selected_features].values
        y = np.where(data_filtered.iloc[:, -1].values == selected_classes[0], -1, 1)

        return X, y

    def initialize_model(self) -> None:
        """Initialize the selected model with current parameters."""
        params = {
            "learning_rate": self.eta_var.get(),
            "epochs": self.epochs_var.get()
        }

        if self.algorithm_var.get() == "Perceptron":
            params["input_dim"] = 2  # We always use 2 features
            self.model = Perceptron(**params)
        else:
            params["mse_threshold"] = self.mse_var.get()
            self.model = Adaline(**params)

    def plot_decision_boundary(self) -> None:
        """Plot the decision boundary and training data."""
        try:
            # Create mesh grid
            x_min, x_max = self.X_train[:, 0].min() - 1, self.X_train[:, 0].max() + 1
            y_min, y_max = self.X_train[:, 1].min() - 1, self.X_train[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                                 np.linspace(y_min, y_max, 200))

            # Predict for all points in the mesh
            Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            # Create plot
            plt.figure(figsize=(8, 6))
            plt.contourf(xx, yy, Z, alpha=0.2, levels=np.linspace(Z.min(), Z.max(), 3), cmap='RdBu')

            # Plot training points
            for class_value, marker, color in zip([-1, 1], ['o', 's'], ['yellow', 'blue']):
                plt.scatter(
                    self.X_train[self.y_train == class_value, 0],
                    self.X_train[self.y_train == class_value, 1],
                    marker=marker, color=color, label=f'Class {class_value}'
                )

            plt.xlabel(self.feature1_var.get())
            plt.ylabel(self.feature2_var.get())
            plt.title("Decision Boundary and Training Data")
            plt.legend()
            plt.show()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot decision boundary: {str(e)}")

    def test_model(self) -> None:
        """
        Tests the trained model on the test set, displays the accuracy, and visualizes the decision boundaries.
        """
        try:
            if self.model is None:
                raise ValueError("Please train the model first.")

            # Make predictions
            y_pred = self.model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            cm = confusion_matrix(self.y_test, y_pred)

            # Display results
            messagebox.showinfo("Results", f"Test Accuracy: {accuracy:.2%}")
            self.plot_confusion_matrix(cm)

        except Exception as e:
            messagebox.showerror("Error", f"Testing failed: {str(e)}")

    def plot_confusion_matrix(self, cm: np.ndarray) -> None:
        """Plot confusion matrix."""
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(cm, cmap='Blues')

        # Add numbers
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center')

        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.colorbar(im)
        plt.show()
