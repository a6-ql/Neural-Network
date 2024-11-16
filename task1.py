import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Main Application Class
class NeuralNetworkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network Classifier")
        self.root.geometry("600x600")

        # Data placeholders
        self.data = None
        self.features = []
        self.class_labels = []

        # Scaler placeholder
        self.scaler = None

        # GUI Components
        self.create_widgets()
    
    def create_widgets(self):
        # Main frame with padding
        frame = ttk.Frame(self.root, padding="20")
        frame.grid(row=0, column=0, sticky="NSEW", padx=20, pady=20)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Load Data Button
        self.load_button = ttk.Button(frame, text="Load Data", command=self.load_data)
        self.load_button.grid(row=0, column=0, columnspan=3, pady=10)

        # Feature selection
        ttk.Label(frame, text="Select Two Features:").grid(row=1, column=0, sticky="w", pady=10, padx=5)
        self.feature1_var = tk.StringVar()
        self.feature2_var = tk.StringVar()
        self.feature1_menu = ttk.Combobox(frame, textvariable=self.feature1_var, width=20, state="readonly")
        self.feature2_menu = ttk.Combobox(frame, textvariable=self.feature2_var, width=20, state="readonly")
        self.feature1_menu.grid(row=1, column=1, pady=10, padx=5)
        self.feature2_menu.grid(row=1, column=2, pady=10, padx=5)
        self.feature1_var.trace("w", self.update_feature_options)
        self.feature2_var.trace("w", self.update_feature_options)

        # Class selection
        ttk.Label(frame, text="Select Two Classes:").grid(row=2, column=0, sticky="w", pady=10, padx=5)
        self.class1_var = tk.StringVar()
        self.class2_var = tk.StringVar()
        self.class1_menu = ttk.Combobox(frame, textvariable=self.class1_var, width=20, state="readonly")
        self.class2_menu = ttk.Combobox(frame, textvariable=self.class2_var, width=20, state="readonly")
        self.class1_menu.grid(row=2, column=1, pady=10, padx=5)
        self.class2_menu.grid(row=2, column=2, pady=10, padx=5)
        self.class1_var.trace("w", self.update_class_options)
        self.class2_var.trace("w", self.update_class_options)

        # Learning rate input
        ttk.Label(frame, text="Learning Rate (eta):").grid(row=3, column=0, sticky="w", pady=10, padx=5)
        self.eta_var = tk.DoubleVar(value=0.01)
        self.eta_entry = ttk.Entry(frame, textvariable=self.eta_var, width=22)
        self.eta_entry.grid(row=3, column=1, pady=10, padx=5)

        # Number of epochs input
        ttk.Label(frame, text="Number of Epochs (m):").grid(row=4, column=0, sticky="w", pady=10, padx=5)
        self.epochs_var = tk.IntVar(value=1000)
        self.epochs_entry = ttk.Entry(frame, textvariable=self.epochs_var, width=22)
        self.epochs_entry.grid(row=4, column=1, pady=10, padx=5)

        # MSE threshold input
        ttk.Label(frame, text="MSE Threshold:").grid(row=5, column=0, sticky="w", pady=10, padx=5)
        self.mse_var = tk.DoubleVar(value=0.001)
        self.mse_entry = ttk.Entry(frame, textvariable=self.mse_var, width=22)
        self.mse_entry.grid(row=5, column=1, pady=10, padx=5)

        # Add Bias checkbox
        self.bias_var = tk.BooleanVar(value=True)
        self.bias_checkbox = ttk.Checkbutton(frame, text="Add Bias", variable=self.bias_var)
        self.bias_checkbox.grid(row=6, column=0, columnspan=3, pady=10)

        # Algorithm selection
        self.algorithm_var = tk.StringVar(value="Perceptron")
        ttk.Label(frame, text="Select Algorithm:").grid(row=7, column=0, sticky="w", pady=10, padx=5)
        ttk.Radiobutton(frame, text="Perceptron", variable=self.algorithm_var, value="Perceptron").grid(row=7, column=1)
        ttk.Radiobutton(frame, text="Adaline", variable=self.algorithm_var, value="Adaline").grid(row=7, column=2)

        # Buttons for training and testing
        self.train_button = ttk.Button(frame, text="Train", command=self.train_model)
        self.train_button.grid(row=8, column=0, pady=20, padx=5)
        self.test_button = ttk.Button(frame, text="Test & Plot", command=self.test_model)
        self.test_button.grid(row=8, column=2, pady=20, padx=5)

    def load_data(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                self.features = self.data.columns[:-1].tolist()
                self.class_labels = self.data.iloc[:, -1].unique().tolist()

                self.feature1_menu["values"] = self.features
                self.feature2_menu["values"] = self.features
                self.class1_menu["values"] = self.class_labels
                self.class2_menu["values"] = self.class_labels

                messagebox.showinfo("Success", "Data loaded successfully and GUI updated!")
                self.preprocess()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load data: {e}")

    def update_feature_options(self, *args):
        if self.features:
            selected_feature1 = self.feature1_var.get()
            selected_feature2 = self.feature2_var.get()

            feature_options = self.features.copy()
            if selected_feature1 in feature_options:
                feature_options.remove(selected_feature1)
            self.feature2_menu["values"] = feature_options

            feature_options = self.features.copy()
            if selected_feature2 in feature_options:
                feature_options.remove(selected_feature2)
            self.feature1_menu["values"] = feature_options

    def update_class_options(self, *args):
        if self.class_labels:
            selected_class1 = self.class1_var.get()
            selected_class2 = self.class2_var.get()

            class_options = self.class_labels.copy()
            if selected_class1 in class_options:
                class_options.remove(selected_class1)
            self.class2_menu["values"] = class_options

            class_options = self.class_labels.copy()
            if selected_class2 in class_options:
                class_options.remove(selected_class2)
            self.class1_menu["values"] = class_options

    def preprocess(self):
        try:
            # Handle missing values in numerical features
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                imputer = SimpleImputer(strategy='mean')
                self.data[numeric_cols] = imputer.fit_transform(self.data[numeric_cols])

            # Encode categorical variables (excluding target variable)
            categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()
            if self.data.columns[-1] in categorical_cols:
                categorical_cols.remove(self.data.columns[-1])  # Exclude target variable
            for column in categorical_cols:
                label_encoder = LabelEncoder()
                self.data[column] = label_encoder.fit_transform(self.data[column])

            # Update class labels after ensuring the target variable remains unchanged
            self.class_labels = self.data.iloc[:, -1].unique().tolist()
            self.class1_menu["values"] = self.class_labels
            self.class2_menu["values"] = self.class_labels

            # Feature Scaling
            self.scaler = StandardScaler()
            feature_cols = self.data.columns[:-1]
            self.data[feature_cols] = self.scaler.fit_transform(self.data[feature_cols])

            messagebox.showinfo("Preprocessing", "Data preprocessing completed successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Data preprocessing failed: {e}")

    def train_model(self):
        try:
            eta = self.eta_var.get()
            epochs = self.epochs_var.get()
            mse_threshold = self.mse_var.get()
            algorithm = self.algorithm_var.get()

            selected_features = [self.feature1_var.get(), self.feature2_var.get()]
            selected_classes = [self.class1_var.get(), self.class2_var.get()]

            if not all(selected_features) or not all(selected_classes):
                raise ValueError("Please select both features and both classes.")

            # Filter the data for the selected classes
            data_filtered = self.data[self.data.iloc[:, -1].isin(selected_classes)]

            if data_filtered.empty:
                raise ValueError("No data available for the selected classes. Please choose different classes.")

            # Map the selected classes to -1 and 1
            y = np.where(data_filtered.iloc[:, -1].values == selected_classes[0], -1, 1)
            X = data_filtered[selected_features].values

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y)

            if X_train.size == 0 or X_test.size == 0:
                raise ValueError("Training or testing set is empty. Please check your data and class selections.")

            self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

            # Initialize the model
            input_dim = X_train.shape[1]
            if algorithm == "Perceptron":
                self.model = Perceptron(input_dim=input_dim, learning_rate=eta, epochs=epochs)
            else:
                self.model = Adaline(learning_rate=eta, epochs=epochs, mse_threshold=mse_threshold)

            # Train the model
            self.model.fit(X_train, y_train)
            messagebox.showinfo("Training", "Model trained successfully!")
            self.plot_decision_boundary()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train model: {e}")

    def plot_decision_boundary(self):
        try:
            # Extract weights and bias from the model
            w = self.model.weights[1:]
            b = self.model.weights[0]

            # Create mesh grid for plotting decision boundary
            x_min, x_max = self.X_train[:, 0].min() - 1, self.X_train[:, 0].max() + 1
            y_min, y_max = self.X_train[:, 1].min() - 1, self.X_train[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                                 np.linspace(y_min, y_max, 200))
            grid = np.c_[xx.ravel(), yy.ravel()]
            Z = self.model.predict(grid)
            Z = Z.reshape(xx.shape)

            # Plot decision boundary and data points
            plt.figure(figsize=(8, 6))
            plt.contourf(xx, yy, Z, alpha=0.2, levels=np.linspace(Z.min(), Z.max(), 3), cmap='RdBu')

            for class_value, marker, color in zip([-1, 1], ['o', 's'], ['blue', 'red']):
                plt.scatter(self.X_train[self.y_train == class_value, 0],
                            self.X_train[self.y_train == class_value, 1],
                            marker=marker, color=color, label=f'Class {class_value}')

            # Labels, legend, and display
            plt.xlabel(self.feature1_var.get())
            plt.ylabel(self.feature2_var.get())
            plt.title("Decision Boundary and Data Points")
            plt.legend()
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot decision boundary: {e}")

    def test_model(self):
        try:
            y_pred = self.model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            cm = confusion_matrix(self.y_test, y_pred)

            messagebox.showinfo("Testing", f"Accuracy: {accuracy:.2f}")

            # Plotting the confusion matrix
            fig, ax = plt.subplots()
            ax.matshow(cm, cmap=plt.cm.Blues)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Confusion Matrix")
            for (i, j), z in np.ndenumerate(cm):
                ax.text(j, i, f'{z}', ha='center', va='center')
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to test model: {e}")

# Define Perceptron and Adaline Models
class Perceptron:
    def __init__(self, input_dim, learning_rate, epochs):
        self.weights = np.zeros(input_dim + 1)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        for epoch in range(self.epochs):
            errors = 0
            for xi, target in zip(X_with_bias, y):
                net_input = np.dot(xi, self.weights)
                prediction = np.where(net_input >= 0.0, 1, -1)
                update = self.learning_rate * (target - prediction)
                self.weights += update * xi
                errors += int(update != 0.0)
            if errors == 0:
                break  # Stop training if no errors

    def predict(self, X):
        X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        net_input = np.dot(X_with_bias, self.weights)
        return np.where(net_input >= 0.0, 1, -1)

class Adaline:
    def __init__(self, learning_rate, epochs, mse_threshold):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.mse_threshold = mse_threshold
        self.weights = None

    def fit(self, X, y):
        X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        self.weights = np.zeros(X_with_bias.shape[1])
        for epoch in range(self.epochs):
            net_input = np.dot(X_with_bias, self.weights)
            errors = y - net_input
            self.weights += self.learning_rate * X_with_bias.T.dot(errors)
            mse = (errors**2).mean()
            if mse < self.mse_threshold:
                break

    def predict(self, X):
        X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        net_input = np.dot(X_with_bias, self.weights)
        return np.where(net_input >= 0.0, 1, -1)

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralNetworkApp(root)
    root.mainloop()
