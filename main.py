import tkinter as tk
from src.gui.app import NeuralNetworkApp

def main():
    """Main entry point of the application."""
    root = tk.Tk()
    app = NeuralNetworkApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()