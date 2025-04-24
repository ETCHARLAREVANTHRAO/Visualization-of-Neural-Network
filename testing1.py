import tkinter as tk
from tkinter import ttk, Canvas, Frame, Label, Button, Scale, StringVar, DoubleVar, IntVar, Text, messagebox, Entry, filedialog
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
import ast

class NeuronLayer:
    def __init__(self, input_size, output_size, activation="sigmoid"):
        if activation in ["sigmoid", "tanh"]:
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / (input_size + output_size))
        else:
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.biases = np.zeros((1, output_size))
        self.activation = activation
        self.input_data = None
        self.output_before_activation = None
        self.output = None

    def forward(self, X):
        self.input_data = X
        if X.ndim != 2:
            raise ValueError(f"Expected 2D input, got shape {X.shape}")
        self.output_before_activation = np.clip(np.dot(X, self.weights) + self.biases, -500, 500)
        if self.activation == "sigmoid":
            self.output = 1 / (1 + np.exp(-self.output_before_activation))
        elif self.activation == "relu":
            self.output = np.maximum(0, self.output_before_activation)
        elif self.activation == "tanh":
            self.output = np.tanh(self.output_before_activation)
        elif self.activation == "linear":
            self.output = self.output_before_activation
        elif self.activation == "softmax":
            z_clipped = np.clip(self.output_before_activation, -500, 500)
            exp_z = np.exp(z_clipped - np.max(z_clipped, axis=1, keepdims=True))
            self.output = exp_z / np.sum(exp_z, axis=1, keepdims=True)
            if not np.allclose(np.sum(self.output, axis=1), 1.0, atol=1e-6):
                raise ValueError("Softmax outputs do not sum to 1")
        return self.output

    def backward(self, d_output, learning_rate):
        if d_output.ndim != 2 or d_output.shape != self.output.shape:
            raise ValueError(f"Expected 2D d_output with shape {self.output.shape}, got {d_output.shape}")
        if self.activation == "sigmoid":
            d_activation = self.output * (1 - self.output)
            delta = d_output * d_activation
        elif self.activation == "relu":
            delta = d_output * (self.output_before_activation > 0).astype(float)
        elif self.activation == "tanh":
            d_activation = 1 - np.square(self.output)
            delta = d_output * d_activation
        elif self.activation == "linear":
            delta = d_output
        elif self.activation == "softmax":
            delta = d_output
        d_weights = np.dot(self.input_data.T, delta)
        d_biases = np.sum(delta, axis=0, keepdims=True)
        d_input = np.dot(delta, self.weights.T)
        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases
        return d_input

class SimpleNeuralNetwork:
    def __init__(self, layer_sizes, activations, loss_function="RMSE"):
        self.layers = [NeuronLayer(layer_sizes[i], layer_sizes[i + 1], activations[i])
                       for i in range(len(layer_sizes) - 1)]
        self.loss_function = loss_function
        self.loss_history = []

    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, X, y, learning_rate):
        output = self.forward(X)
        if self.loss_function == "RMSE":
            loss = np.mean(np.square(output - y))
            d_output = 2 * (output - y) / y.shape[0]
        elif self.loss_function == "BCE":
            output = np.clip(output, 1e-15, 1 - 1e-15)
            loss = -np.mean(y * np.log(output) + (1 - y) * np.log(1 - output))
            d_output = -(y / output - (1 - y) / (1 - output)) / y.shape[0]
        elif self.loss_function == "CCE":
            output = np.clip(output, 1e-15, 1 - 1e-15)
            loss = -np.mean(np.sum(y * np.log(output), axis=1))
            d_output = (output - y) / y.shape[0]
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_function}")
        self.loss_history.append(loss)
        for layer in reversed(self.layers):
            d_output = layer.backward(d_output, learning_rate)
        return loss

class NeuralNetworkVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Layer Neural Network Visualization")
        self.root.geometry("1200x700")
        self.setup_network_params()
        self.create_frames()
        self.setup_control_panel()
        self.setup_network_visualization()
        self.setup_plots()
        self.is_animating = False
        self.current_epoch = 0
        self.animation = None
        self.root.bind("<Configure>", self.on_resize)

    def setup_network_params(self):
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y = np.array([[0], [1], [1], [0]])
        self.input_size = self.X.shape[1]
        self.output_size = self.y.shape[1]
        self.hidden_layers = [8, 4]
        self.layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        self.activations = ["sigmoid"] * (len(self.hidden_layers) + 1)
        self.loss_function = "RMSE"
        self.network = SimpleNeuralNetwork(self.layer_sizes, self.activations, self.loss_function)
        self.learning_rate = 0.1

    def create_frames(self):
        self.control_frame = Frame(self.root, bg="#e0e0e0", width=250)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        self.control_frame.pack_propagate(False)
        self.main_frame = Frame(self.root, bg="#f5f5f5")
        self.main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.network_frame = Frame(self.main_frame, bg="#ffffff", height=400)
        self.network_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.plots_frame = Frame(self.main_frame, bg="#ffffff", height=300)
        self.plots_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.loss_plot_frame = Frame(self.plots_frame, bg="#ffffff")
        self.loss_plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.activation_plot_frame = Frame(self.plots_frame, bg="#ffffff")
        self.activation_plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

    def setup_control_panel(self):
        Label(self.control_frame, text="Control Panel", font=("Arial", 12, "bold"), bg="#e0e0e0").pack(pady=5)
        Label(self.control_frame, text="Network Architecture", font=("Arial", 10, "bold"), bg="#e0e0e0").pack(pady=2)
        Label(self.control_frame, text="Hidden Layers (e.g., 8,4,2):", bg="#e0e0e0").pack()
        self.hidden_layers_var = StringVar(value="8,4")
        Entry(self.control_frame, textvariable=self.hidden_layers_var).pack(fill=tk.X, padx=5, pady=2)
        Label(self.control_frame, text="Activations (e.g., relu,sigmoid,softmax):", bg="#e0e0e0").pack()
        self.activations_var = StringVar(value="sigmoid,sigmoid,sigmoid")
        Entry(self.control_frame, textvariable=self.activations_var).pack(fill=tk.X, padx=5, pady=2)
        Label(self.control_frame, text="Options: sigmoid, relu, tanh, linear, softmax", bg="#e0e0e0", font=("Arial", 8)).pack()
        Label(self.control_frame, text="Loss Function:", bg="#e0e0e0").pack()
        self.loss_var = StringVar(value="RMSE")
        ttk.Combobox(self.control_frame, textvariable=self.loss_var,
                     values=["RMSE", "BCE", "CCE"], state="readonly").pack(fill=tk.X, padx=5)
        Label(self.control_frame, text="Dataset", font=("Arial", 10, "bold"), bg="#e0e0e0").pack(pady=2)
        self.dataset_var = StringVar(value="XOR")
        ttk.Combobox(self.control_frame, textvariable=self.dataset_var,
                     values=["XOR", "AND", "OR", "NAND", "NOR", "XNOR", "Circles", "Moons", "Iris", "Custom"], state="readonly").pack(fill=tk.X, padx=5)
        self.dataset_var.trace("w", self.change_dataset)
        Label(self.control_frame, text="Custom Input (e.g., [[0,0],[0,1],[1,0],[1,1]]):", bg="#e0e0e0").pack()
        self.input_text = Text(self.control_frame, height=2, width=20)
        self.input_text.pack(fill=tk.X, padx=5)
        Label(self.control_frame, text="Custom Output (e.g., [[0],[1],[1],[0]] for BCE/RMSE):", bg="#e0e0e0").pack()
        self.output_text = Text(self.control_frame, height=2, width=20)
        self.output_text.pack(fill=tk.X, padx=5)
        Button(self.control_frame, text="Load CSV", command=self.load_csv_dataset,
               bg="#ff5722", fg="white").pack(fill=tk.X, padx=5, pady=5)
        Label(self.control_frame, text="Training Parameters", font=("Arial", 10, "bold"), bg="#e0e0e0").pack(pady=2)
        Label(self.control_frame, text="Learning Rate:", bg="#e0e0e0").pack()
        self.learning_rate_var = DoubleVar(value=0.1)
        Scale(self.control_frame, from_=0.01, to=0.5, resolution=0.01,
              orient=tk.HORIZONTAL, variable=self.learning_rate_var, bg="#e0e0e0").pack(fill=tk.X)
        Label(self.control_frame, text="Animation Speed:", bg="#e0e0e0").pack()
        self.speed_var = IntVar(value=50)
        Scale(self.control_frame, from_=1, to=100, orient=tk.HORIZONTAL,
              variable=self.speed_var, bg="#e0e0e0").pack(fill=tk.X)
        button_frame = Frame(self.control_frame, bg="#e0e0e0")
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        self.init_button = Button(button_frame, text="Initialize Network", command=self.initialize_network,
                                  bg="#4caf50", fg="white")
        self.init_button.grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        self.anim_button = Button(button_frame, text="Start Training", command=self.toggle_animation,
                                  bg="#2196f3", fg="white")
        self.anim_button.grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        self.step_button = Button(button_frame, text="Step Epoch", command=self.step_epoch,
                                  bg="#ff9800", fg="white")
        self.step_button.grid(row=1, column=0, sticky="ew", padx=2, pady=2)
        self.reset_button = Button(button_frame, text="Reset", command=self.reset_visualization,
                                   bg="#f44336", fg="white")
        self.reset_button.grid(row=1, column=1, sticky="ew", padx=2, pady=2)
        self.test_button = Button(button_frame, text="Test Network", command=self.test_network,
                                  bg="#9c27b0", fg="white")
        self.test_button.grid(row=2, column=0, sticky="ew", padx=2, pady=2)
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)
        self.status_var = StringVar(value="Ready")
        Label(self.control_frame, textvariable=self.status_var,
              bg="#ffffff", relief=tk.SUNKEN).pack(fill=tk.X, pady=5)
        self.epoch_var = StringVar(value="Epoch: 0")
        Label(self.control_frame, textvariable=self.epoch_var,
              bg="#ffffff", relief=tk.SUNKEN).pack(fill=tk.X, pady=5)
        self.loss_var_display = StringVar(value="Loss: N/A")
        Label(self.control_frame, textvariable=self.loss_var_display,
              bg="#ffffff", relief=tk.SUNKEN, font=("Arial", 12, "bold")).pack(fill=tk.X, pady=5)

    def load_csv_dataset(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        try:
            df = pd.read_csv(file_path)
            if df.empty or len(df.columns) < 2:
                raise ValueError("CSV must have at least 2 columns (features + output)")
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values.reshape(-1, 1)
            if X.ndim != 2 or y.ndim != 2 or X.shape[0] != y.shape[0]:
                raise ValueError("Invalid CSV format: X and y must be 2D with same number of samples")
            self.X = X
            self.y = y
            self.input_size = self.X.shape[1]
            self.output_size = self.y.shape[1]
            if self.loss_var.get() == "BCE":
                if not np.all(np.isin(self.y, [0, 1])) or self.output_size != 1:
                    raise ValueError("BCE requires binary outputs (0 or 1) and a single output neuron")
            self.update_layer_sizes()
            self.initialize_network()
            self.dataset_var.set("Custom")
            self.status_var.set(f"Loaded CSV dataset: {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV: {str(e)}")
            self.dataset_var.set("XOR")
            self.change_dataset()

    def change_dataset(self, *args):
        dataset = self.dataset_var.get()
        try:
            if dataset == "Custom":
                input_str = self.input_text.get("1.0", tk.END).strip()
                output_str = self.output_text.get("1.0", tk.END).strip()
                self.X = np.array(ast.literal_eval(input_str))
                self.y = np.array(ast.literal_eval(output_str))
                if self.X.ndim != 2 or self.y.ndim != 2 or self.X.shape[0] != self.y.shape[0]:
                    raise ValueError("Invalid input/output dimensions: X and y must be 2D with same number of samples")
            else:
                if dataset == "XOR":
                    self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
                    self.y = np.array([[0], [1], [1], [0]])
                elif dataset == "AND":
                    self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
                    self.y = np.array([[0], [0], [0], [1]])
                elif dataset == "OR":
                    self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
                    self.y = np.array([[0], [1], [1], [1]])
                elif dataset == "NAND":
                    self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
                    self.y = np.array([[1], [1], [1], [0]])
                elif dataset == "NOR":
                    self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
                    self.y = np.array([[1], [0], [0], [0]])
                elif dataset == "XNOR":
                    self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
                    self.y = np.array([[1], [0], [0], [1]])
                elif dataset == "Circles":
                    self.X = np.array([
                        [0.45, -0.12], [0.67, 0.23], [-0.15, -0.60], [0.10, 0.78],
                        [-0.67, 0.23], [0.55, 0.45], [-0.45, 0.12], [0.15, 0.60],
                        [0.87, -0.34], [-0.87, 0.34]
                    ])
                    self.y = np.array([[1], [0], [0], [1], [0], [1], [1], [0], [1], [0]])
                elif dataset == "Moons":
                    self.X = np.array([
                        [0.87, 0.34], [1.12, -0.45], [0.25, 0.90], [0.75, -0.10],
                        [1.45, -0.20], [0.55, 0.45], [1.67, -0.23], [0.33, 0.67],
                        [1.23, -0.12], [0.45, 0.78]
                    ])
                    self.y = np.array([[0], [1], [0], [1], [1], [0], [1], [0], [1], [0]])
                elif dataset == "Iris":
                    self.X = np.array([
                        [0.22, 0.62, 0.07, 0.04], [0.44, 0.54, 0.23, 0.10],
                        [0.36, 0.50, 0.18, 0.08], [0.61, 0.58, 0.45, 0.25],
                        [0.67, 0.69, 0.56, 0.29], [0.36, 0.42, 0.13, 0.06],
                        [0.56, 0.50, 0.40, 0.25], [0.75, 0.73, 0.63, 0.33],
                        [0.28, 0.58, 0.10, 0.06], [0.64, 0.62, 0.49, 0.25]
                    ])
                    self.y = np.array([[0], [0], [0], [1], [1], [0], [1], [1], [0], [1]])
            self.input_size = self.X.shape[1]
            self.output_size = self.y.shape[1]
            if self.loss_var.get() == "BCE":
                if not np.all(np.isin(self.y, [0, 1])) or self.output_size != 1:
                    raise ValueError("BCE requires binary outputs (0 or 1) and a single output neuron")
            elif self.loss_var.get() == "CCE":
                if not np.all(np.isin(self.y, [0, 1])):
                    raise ValueError("CCE requires binary (0 or 1) values in one-hot encoded outputs")
                if np.any(np.sum(self.y, axis=1) != 1):
                    raise ValueError("CCE requires one-hot encoded outputs where each row sums to 1")
                if self.y.shape[1] < 2:
                    raise ValueError("CCE requires at least 2 output classes for softmax")
            self.update_layer_sizes()
            self.initialize_network()
            self.status_var.set(f"Loaded {dataset} dataset")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid dataset format: {str(e)}")
            self.dataset_var.set("XOR")
            self.change_dataset()

    def update_layer_sizes(self):
        try:
            hidden_str = self.hidden_layers_var.get().strip()
            if hidden_str:
                self.hidden_layers = [int(x) for x in hidden_str.split(',')]
            else:
                self.hidden_layers = []
            self.layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
            activations_str = self.activations_var.get().strip()
            if activations_str:
                self.activations = [x.strip() for x in activations_str.split(',')]
            else:
                self.activations = ["sigmoid"] * (len(self.layer_sizes) - 1)
            expected_activations = len(self.layer_sizes) - 1
            if len(self.activations) != expected_activations:
                self.activations = ["sigmoid"] * expected_activations
                self.activations_var.set(','.join(self.activations))
                messagebox.showwarning("Warning", f"Adjusted activations to match {expected_activations} layers")
            valid_activations = ["sigmoid", "relu", "tanh", "linear", "softmax"]
            if not all(act in valid_activations for act in self.activations):
                raise ValueError(f"Invalid activation function. Use: {', '.join(valid_activations)}")
            if self.loss_var.get() == "CCE" and self.activations[-1] != "softmax":
                self.activations[-1] = "softmax"
                self.activations_var.set(','.join(self.activations))
                messagebox.showinfo("Info", "Output layer activation set to softmax for CCE loss")
            if self.loss_var.get() == "BCE" and self.output_size == 1 and self.activations[-1] != "sigmoid":
                self.activations[-1] = "sigmoid"
                self.activations_var.set(','.join(self.activations))
                messagebox.showinfo("Info", "Output layer activation set to sigmoid for BCE")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid configuration: {str(e)}")
            self.hidden_layers = [8, 4]
            self.hidden_layers_var.set("8,4")
            self.layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
            self.activations = ["sigmoid"] * (len(self.layer_sizes) - 1)
            self.activations_var.set(','.join(self.activations))

    def setup_network_visualization(self):
        self.canvas = Canvas(self.network_frame, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.draw_network()

    def setup_plots(self):
        self.loss_figure = Figure(figsize=(5, 4))
        self.loss_axes = self.loss_figure.add_subplot(111)
        self.loss_axes.set_title("Loss Over Epochs", fontsize=12)
        self.loss_axes.set_xlabel("Epoch")
        self.loss_axes.set_ylabel("Loss")
        self.loss_axes.grid(True)
        self.loss_canvas = FigureCanvasTkAgg(self.loss_figure, self.loss_plot_frame)
        self.loss_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.activation_figure = Figure(figsize=(5, 4))
        self.activation_axes = self.activation_figure.add_subplot(111)
        self.activation_axes.set_title("Activation", fontsize=12)
        self.activation_axes.grid(True)
        self.activation_canvas = FigureCanvasTkAgg(self.activation_figure, self.activation_plot_frame)
        self.activation_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.update_plots()

    def on_resize(self, event):
        self.draw_network()
        self.update_network_visualization()

    def draw_network(self):
        self.canvas.delete("all")
        width = self.canvas.winfo_width() or 800
        height = self.canvas.winfo_height() or 400
        h_spacing = width / (len(self.layer_sizes) + 1)
        self.neuron_positions = []
        for l, layer_size in enumerate(self.layer_sizes):
            layer_positions = []
            v_spacing = height / (layer_size + 1)
            layer_label = "Input Layer" if l == 0 else "Output Layer" if l == len(self.layer_sizes) - 1 else f"Hidden Layer {l}"
            self.canvas.create_text((l + 1) * h_spacing, 20, text=layer_label, font=("Arial", 10, "bold"))
            for n in range(layer_size):
                x = (l + 1) * h_spacing
                y = (n + 1) * v_spacing
                layer_positions.append((x, y))
                color = "#3498db" if l == 0 else "#e74c3c" if l == len(self.layer_sizes) - 1 else "#2ecc71"
                self.canvas.create_oval(x - 15, y - 15, x + 15, y + 15, fill=color, outline="black", tags=f"neuron_{l}_{n}")
                self.canvas.create_text(x, y, text="0.0", fill="white", tags=f"value_{l}_{n}")
                if l > 0:
                    self.canvas.create_text(x, y + 20, text="z: 0.0", font=("Arial", 8), tags=f"preact_{l}_{n}")
                    self.canvas.create_text(x - 30, y - 10, text="b: 0.0", font=("Arial", 8), tags=f"bias_{l}_{n}")
            self.neuron_positions.append(layer_positions)
        if not self.network.layers:
            return
        for l in range(len(self.layer_sizes) - 1):
            for i, pos1 in enumerate(self.neuron_positions[l]):
                for j, pos2 in enumerate(self.neuron_positions[l + 1]):
                    weight = self.network.layers[l].weights[i, j]
                    line_width = min(abs(weight) * 2 + 0.5, 5)
                    line_color = "#2ecc71" if weight >= 0 else "#e74c3c"
                    self.canvas.create_line(pos1[0], pos1[1], pos2[0], pos2[1],
                                            fill=line_color, width=line_width, tags=f"connection_{l}_{i}_{j}")
                    mid_x = (pos1[0] + pos2[0]) / 2
                    mid_y = (pos1[1] + pos2[1]) / 2
                    self.canvas.create_text(mid_x, mid_y, text=f"{weight:.2f}", font=("Arial", 8), tags=f"weight_{l}_{i}_{j}")

    def update_network_visualization(self):
        if not self.network.layers:
            return
        for l in range(len(self.layer_sizes) - 1):
            for i in range(self.layer_sizes[l]):
                for j in range(self.layer_sizes[l + 1]):
                    weight = self.network.layers[l].weights[i, j]
                    line_width = min(abs(weight) * 2 + 0.5, 5)
                    line_color = "#2ecc71" if weight >= 0 else "#e74c3c"
                    self.canvas.itemconfig(f"connection_{l}_{i}_{j}", width=line_width, fill=line_color)
                    self.canvas.itemconfig(f"weight_{l}_{i}_{j}", text=f"{weight:.2f}")
            for j in range(self.layer_sizes[l + 1]):
                bias = self.network.layers[l].biases[0, j]
                self.canvas.itemconfig(f"bias_{l + 1}_{j}", text=f"b: {bias:.2f}")
        output = self.network.forward(self.X[:1])
        for i in range(self.layer_sizes[0]):
            self.canvas.itemconfig(f"value_0_{i}", text=f"{self.X[0, i]:.2f}")
        for l in range(len(self.network.layers)):
            for j in range(self.layer_sizes[l + 1]):
                activation = self.network.layers[l].output[0, j]
                preact = self.network.layers[l].output_before_activation[0, j]
                self.canvas.itemconfig(f"value_{l + 1}_{j}", text=f"{activation:.2f}")
                self.canvas.itemconfig(f"preact_{l + 1}_{j}", text=f"z: {preact:.2f}")

    def update_plots(self):
        self.loss_axes.clear()
        self.loss_axes.set_title("Loss Over Epochs", fontsize=12)
        self.loss_axes.set_xlabel("Epoch")
        self.loss_axes.set_ylabel("Loss")
        self.loss_axes.grid(True)
        if self.network.loss_history:
            self.loss_axes.plot(self.network.loss_history, 'b-', label="Loss")
            self.loss_axes.legend()
            self.loss_var_display.set(f"Loss: {self.network.loss_history[-1]:.6f}")
        self.loss_canvas.draw()
        self.activation_axes.clear()
        self.activation_axes.set_title(f"Output Layer Activation: {self.activations[-1].capitalize()}", fontsize=12)
        x = np.linspace(-5, 5, 100)
        if self.activations[-1] == "sigmoid":
            y = 1 / (1 + np.exp(-x))
            derivative = y * (1 - y)
            formula = r"$f(x) = \frac{1}{1 + e^{-x}}$"
            self.activation_axes.plot(x, y, 'b-', label='Activation')
            self.activation_axes.plot(x, derivative, 'r--', label='Derivative')
        elif self.activations[-1] == "relu":
            y = np.maximum(0, x)
            derivative = np.where(x > 0, 1, 0)
            formula = r"$f(x) = \max(0, x)$"
            self.activation_axes.plot(x, y, 'b-', label='Activation')
            self.activation_axes.plot(x, derivative, 'r--', label='Derivative')
        elif self.activations[-1] == "tanh":
            y = np.tanh(x)
            derivative = 1 - y ** 2
            formula = r"$f(x) = \tanh(x)$"
            self.activation_axes.plot(x, y, 'b-', label='Activation')
            self.activation_axes.plot(x, derivative, 'r--', label='Derivative')
        elif self.activations[-1] == "linear":
            y = x
            derivative = np.ones_like(x)
            formula = r"$f(x) = x$"
            self.activation_axes.plot(x, y, 'b-', label='Activation')
            self.activation_axes.plot(x, derivative, 'r--', label='Derivative')
        elif self.activations[-1] == "softmax":
            x_vec = np.array([x, -x, x/2]).T
            exp_x = np.exp(x_vec - np.max(x_vec, axis=1, keepdims=True))
            y = exp_x / np.sum(exp_x, axis=1, keepdims=True)
            formula = r"$f(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$"
            for i in range(3):
                self.activation_axes.plot(x, y[:, i], label=f'Class {i+1}')
            self.activation_axes.text(0, -0.5, "Note: Shows softmax for 3-class example", fontsize=8)
        self.activation_axes.legend()
        self.activation_axes.text(0, -0.3, formula, fontsize=12)
        self.activation_axes.grid(True)
        self.activation_canvas.draw()

    def initialize_network(self):
        self.update_layer_sizes()
        self.loss_function = self.loss_var.get()
        self.network = SimpleNeuralNetwork(self.layer_sizes, self.activations, self.loss_function)
        self.current_epoch = 0
        self.epoch_var.set("Epoch: 0")
        self.loss_var_display.set("Loss: N/A")
        self.network.loss_history = []
        self.draw_network()
        self.update_network_visualization()
        self.update_plots()
        self.status_var.set("Network Initialized")

    def toggle_animation(self):
        if self.is_animating:
            self.is_animating = False
            if self.animation:
                self.animation.event_source.stop()
                self.animation = None
            self.anim_button.config(text="Start Training", bg="#2196f3")
            self.status_var.set("Training Stopped")
        else:
            self.is_animating = True
            self.anim_button.config(text="Stop Training", bg="#f44336")
            self.status_var.set("Training in Progress...")
            self.animation = FuncAnimation(self.loss_figure, self.animate,
                                          interval=5000 / self.speed_var.get(), blit=False)
            self.loss_canvas.draw()

    def step_epoch(self):
        if self.is_animating:
            messagebox.showwarning("Warning", "Stop animation before stepping manually")
            return
        self.perform_epoch()

    def perform_epoch(self):
        indices = np.random.permutation(self.X.shape[0])
        X_shuffled = self.X[indices]
        y_shuffled = self.y[indices]
        loss = self.network.backward(
            X_shuffled,
            y_shuffled,
            self.learning_rate_var.get()
        )
        self.current_epoch += 1
        self.epoch_var.set(f"Epoch: {self.current_epoch}")
        self.loss_var_display.set(f"Loss: {loss:.6f}")
        self.update_network_visualization()
        self.update_plots()

    def animate(self, frame):
        self.perform_epoch()

    def test_network(self):
        predictions = self.network.forward(self.X)
        result_text = "Training Results:\n"
        result_text += "-" * 50 + "\n"
        result_text += f"{'Input':^30}{'Expected':^15}{'Predicted':^15}\n"
        result_text += "-" * 50 + "\n"
        correct_count = 0
        for i in range(self.X.shape[0]):
            predicted = predictions[i]
            expected = self.y[i]
            if self.loss_function == "CCE":
                is_correct = np.argmax(predicted) == np.argmax(expected)
                predicted_display = np.round(predicted, 4)
            elif self.loss_function == "BCE":
                predicted_binary = (predicted > 0.5).astype(int)
                is_correct = np.all(predicted_binary == expected)
                predicted_display = predicted_binary
            else:
                if np.all(np.isin(self.y, [0, 1])) and self.output_size == 1:
                    threshold = 0.5 if self.activations[-1] == "sigmoid" else (np.max(predicted) + np.min(predicted)) / 2
                    predicted_binary = (predicted > threshold).astype(int)
                    is_correct = np.all(predicted_binary == expected)
                    predicted_display = predicted_binary
                else:
                    is_correct = np.allclose(predicted, expected, atol=0.1)
                    predicted_display = np.round(predicted, 4)
            if is_correct:
                correct_count += 1
            result_text += f"{str(self.X[i]):^30}{str(expected):^15}{str(predicted_display):^15}\n"
        result_text += "-" * 50 + "\n"
        accuracy = (correct_count / self.X.shape[0]) * 100
        result_text += f"\nAccuracy: {accuracy:.2f}% ({correct_count}/{self.X.shape[0]} correct)"
        messagebox.showinfo("Network Test Results", result_text)

    def reset_visualization(self):
        if self.is_animating:
            self.toggle_animation()
        self.initialize_network()
        self.status_var.set("Network Reset")

if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralNetworkVisualizer(root)
    root.mainloop()
