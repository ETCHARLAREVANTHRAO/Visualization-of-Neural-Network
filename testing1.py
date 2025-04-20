import tkinter as tk
from tkinter import ttk, Canvas, Frame, Label, Button, Scale, StringVar, DoubleVar, IntVar, Text, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
import ast

class NeuronLayer:
    def __init__(self, input_size, output_size, activation="sigmoid"):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.biases = np.random.randn(1, output_size) * 0.1
        self.activation = activation
        self.input_data = None
        self.output_before_activation = None
        self.output = None

    def forward(self, X):
        self.input_data = X
        self.output_before_activation = np.dot(X, self.weights) + self.biases
        if self.activation == "sigmoid":
            self.output = 1 / (1 + np.exp(-self.output_before_activation))
        elif self.activation == "relu":
            self.output = np.maximum(0, self.output_before_activation)
        elif self.activation == "tanh":
            self.output = np.tanh(self.output_before_activation)
        elif self.activation == "linear":
            self.output = self.output_before_activation
        return self.output

    def backward(self, d_output, learning_rate):
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
        else:  # BCE
            output = np.clip(output, 1e-15, 1 - 1e-15)  # Avoid log(0)
            loss = -np.mean(y * np.log(output) + (1 - y) * np.log(1 - output))
            d_output = -(y / output - (1 - y) / (1 - output)) / y.shape[0]
        self.loss_history.append(loss)
        for layer in reversed(self.layers):
            d_output = layer.backward(d_output, learning_rate)
        return loss

class NeuralNetworkVisualizer1:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network Visualization")
        self.root.geometry("1000x600")
        self.setup_network_params()
        self.create_frames()
        self.setup_control_panel()
        self.setup_network_visualization()
        self.setup_plots()
        self.is_animating = False
        self.current_epoch = 0
        self.max_epochs = 100
        self.animation = None

    def setup_network_params(self):
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y = np.array([[0], [1], [1], [0]])
        self.layer_sizes = [2, 4, 1]
        self.activations = ["sigmoid", "sigmoid"]
        self.loss_function = "RMSE"
        self.network = SimpleNeuralNetwork(self.layer_sizes, self.activations, self.loss_function)
        self.learning_rate = 0.1
        self.batch_size = 4

    def create_frames(self):
        self.control_frame = Frame(self.root, bg="#e0e0e0", width=200)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        self.control_frame.pack_propagate(False)
        self.main_frame = Frame(self.root, bg="#f5f5f5")
        self.main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.network_frame = Frame(self.main_frame, bg="#ffffff", height=300)
        self.network_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.plots_frame = Frame(self.main_frame, bg="#ffffff", height=300)
        self.plots_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.loss_plot_frame = Frame(self.plots_frame, bg="#ffffff")
        self.loss_plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.activation_plot_frame = Frame(self.plots_frame, bg="#ffffff")
        self.activation_plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

    def setup_control_panel(self):
        Label(self.control_frame, text="Control Panel", font=("Arial", 12, "bold"), bg="#e0e0e0").pack(pady=5)
        Label(self.control_frame, text="Hidden Neurons:", bg="#e0e0e0").pack()
        self.hidden_neurons_var = IntVar(value=4)
        Scale(self.control_frame, from_=2, to=16, orient=tk.HORIZONTAL, variable=self.hidden_neurons_var, bg="#e0e0e0").pack(fill=tk.X)
        Label(self.control_frame, text="Activation:", bg="#e0e0e0").pack()
        self.activation_var = StringVar(value="sigmoid")
        ttk.Combobox(self.control_frame, textvariable=self.activation_var, values=["sigmoid", "relu", "tanh", "linear"], state="readonly").pack(fill=tk.X, padx=5)
        Label(self.control_frame, text="Loss Function:", bg="#e0e0e0").pack()
        self.loss_var = StringVar(value="RMSE")
        ttk.Combobox(self.control_frame, textvariable=self.loss_var, values=["RMSE", "BCE"], state="readonly").pack(fill=tk.X, padx=5)
        Label(self.control_frame, text="Dataset:", bg="#e0e0e0").pack()
        self.dataset_var = StringVar(value="XOR")
        ttk.Combobox(self.control_frame, textvariable=self.dataset_var, values=["XOR", "AND", "OR", "Custom"], state="readonly").pack(fill=tk.X, padx=5)
        self.dataset_var.trace("w", self.change_dataset)
        Label(self.control_frame, text="Custom Input (e.g., [[0,0],[0,1],[1,0],[1,1]]):", bg="#e0e0e0").pack()
        self.input_text = Text(self.control_frame, height=2, width=20)
        self.input_text.pack(fill=tk.X, padx=5)
        Label(self.control_frame, text="Custom Output (e.g., [[0],[1],[1],[0]]):", bg="#e0e0e0").pack()
        self.output_text = Text(self.control_frame, height=2, width=20)
        self.output_text.pack(fill=tk.X, padx=5)
        Label(self.control_frame, text="Learning Rate:", bg="#e0e0e0").pack()
        self.learning_rate_var = DoubleVar(value=0.1)
        Scale(self.control_frame, from_=0.01, to=0.5, resolution=0.01, orient=tk.HORIZONTAL, variable=self.learning_rate_var, bg="#e0e0e0").pack(fill=tk.X)
        Label(self.control_frame, text="Animation Speed:", bg="#e0e0e0").pack()
        self.speed_var = IntVar(value=50)
        Scale(self.control_frame, from_=1, to=100, orient=tk.HORIZONTAL, variable=self.speed_var, bg="#e0e0e0").pack(fill=tk.X)
        Button(self.control_frame, text="Initialize", command=self.initialize_network, bg="#4caf50", fg="white").pack(fill=tk.X, pady=5)
        self.anim_button = Button(self.control_frame, text="Start Training", command=self.toggle_animation, bg="#2196f3", fg="white")
        self.anim_button.pack(fill=tk.X, pady=5)
        Button(self.control_frame, text="Reset", command=self.reset_visualization, bg="#f44336", fg="white").pack(fill=tk.X, pady=5)
        self.status_var = StringVar(value="Ready")
        Label(self.control_frame, textvariable=self.status_var, bg="#ffffff", relief=tk.SUNKEN).pack(fill=tk.X, pady=5)
        self.epoch_var = StringVar(value="Epoch: 0")
        Label(self.control_frame, textvariable=self.epoch_var, bg="#ffffff", relief=tk.SUNKEN).pack(fill=tk.X, pady=5)
        self.loss_var_display = StringVar(value="Loss: N/A")
        Label(self.control_frame, textvariable=self.loss_var_display, bg="#ffffff", relief=tk.SUNKEN).pack(fill=tk.X, pady=5)

    def change_dataset(self, *args):
        dataset = self.dataset_var.get()
        try:
            if dataset == "Custom":
                input_str = self.input_text.get("1.0", tk.END).strip()
                output_str = self.output_text.get("1.0", tk.END).strip()
                self.X = np.array(ast.literal_eval(input_str))
                self.y = np.array(ast.literal_eval(output_str))
                if self.X.shape[0] != self.y.shape[0] or self.X.shape[1] != 2 or self.y.shape[1] != 1:
                    raise ValueError("Invalid input/output dimensions")
            else:
                self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
                if dataset == "XOR":
                    self.y = np.array([[0], [1], [1], [0]])
                elif dataset == "AND":
                    self.y = np.array([[0], [0], [0], [1]])
                elif dataset == "OR":
                    self.y = np.array([[0], [1], [1], [1]])
            self.layer_sizes = [self.X.shape[1], self.hidden_neurons_var.get(), 1]
            self.initialize_network()
            self.status_var.set(f"Loaded {dataset} dataset")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid dataset format: {str(e)}")
            self.dataset_var.set("XOR")
            self.change_dataset()

    def setup_network_visualization(self):
        self.canvas = Canvas(self.network_frame, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.draw_network()

    def setup_plots(self):
        self.loss_figure = Figure(figsize=(4, 3))
        self.loss_axes = self.loss_figure.add_subplot(111)
        self.loss_axes.set_title("Loss")
        self.loss_axes.grid(True)
        self.loss_canvas = FigureCanvasTkAgg(self.loss_figure, self.loss_plot_frame)
        self.loss_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.activation_figure = Figure(figsize=(4, 3))
        self.activation_axes = self.activation_figure.add_subplot(111)
        self.activation_axes.set_title("Activation")
        self.activation_axes.grid(True)
        self.activation_canvas = FigureCanvasTkAgg(self.activation_figure, self.activation_plot_frame)
        self.activation_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.update_plots()

    def draw_network(self):
        self.canvas.delete("all")
        width = self.canvas.winfo_width() or 600
        height = self.canvas.winfo_height() or 300
        h_spacing = width / (len(self.layer_sizes) + 1)
        self.neuron_positions = []
        for l, layer_size in enumerate(self.layer_sizes):
            layer_positions = []
            v_spacing = height / (layer_size + 1)
            for n in range(layer_size):
                x = (l + 1) * h_spacing
                y = (n + 1) * v_spacing
                layer_positions.append((x, y))
                color = "#3498db" if l < len(self.layer_sizes) - 1 else "#e74c3c"
                self.canvas.create_oval(x - 15, y - 15, x + 15, y + 15, fill=color, outline="black", tags=f"neuron_{l}_{n}")
                self.canvas.create_text(x, y, text="0.0", fill="white", tags=f"value_{l}_{n}")
                self.canvas.create_text(x, y + 20, text="z: 0.0", font=("Arial", 8), tags=f"preact_{l}_{n}")
                if l > 0:
                    self.canvas.create_text(x - 30, y - 10, text="b: 0.0", font=("Arial", 8), tags=f"bias_{l}_{n}")
            self.neuron_positions.append(layer_positions)
        for l in range(len(self.layer_sizes) - 1):
            for i, pos1 in enumerate(self.neuron_positions[l]):
                for j, pos2 in enumerate(self.neuron_positions[l + 1]):
                    weight = self.network.layers[l].weights[i, j] if self.network.layers else 0
                    line_width = abs(weight) * 2 + 0.5
                    line_color = "#2ecc71" if weight >= 0 else "#e74c3c"
                    self.canvas.create_line(pos1[0], pos1[1], pos2[0], pos2[1], fill=line_color, width=line_width, tags=f"connection_{l}_{i}_{j}")
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
                    line_width = abs(weight) * 2 + 0.5
                    line_color = "#2ecc71" if weight >= 0 else "#e74c3c"
                    self.canvas.itemconfig(f"connection_{l}_{i}_{j}", width=line_width, fill=line_color)
                    self.canvas.itemconfig(f"weight_{l}_{i}_{j}", text=f"{weight:.2f}")
            for j in range(self.layer_sizes[l + 1]):
                bias = self.network.layers[l].biases[0, j]
                self.canvas.itemconfig(f"bias_{l + 1}_{j}", text=f"b: {bias:.2f}")
        output = self.network.forward(self.X)
        for i in range(self.layer_sizes[0]):
            self.canvas.itemconfig(f"value_0_{i}", text=f"{self.X[0, i]:.2f}")
            self.canvas.itemconfig(f"preact_0_{i}", text=f"z: {self.X[0, i]:.2f}")
        for l in range(len(self.network.layers)):
            for j in range(self.layer_sizes[l + 1]):
                activation = self.network.layers[l].output[0, j]
                preact = self.network.layers[l].output_before_activation[0, j]
                self.canvas.itemconfig(f"value_{l + 1}_{j}", text=f"{activation:.2f}")
                self.canvas.itemconfig(f"preact_{l + 1}_{j}", text=f"z: {preact:.2f}")

    def update_plots(self):
        self.loss_axes.clear()
        self.loss_axes.set_title("Loss")
        self.loss_axes.grid(True)
        if self.network.loss_history:
            self.loss_axes.plot(self.network.loss_history, 'b-')
            self.loss_var_display.set(f"Loss: {self.network.loss_history[-1]:.6f}")
        self.loss_canvas.draw()
        self.activation_axes.clear()
        self.activation_axes.set_title(f"{self.activation_var.get().capitalize()}")
        x = np.linspace(-5, 5, 100)
        if self.activation_var.get() == "sigmoid":
            y = 1 / (1 + np.exp(-x))
            derivative = y * (1 - y)
            formula = r"$f(x) = \frac{1}{1 + e^{-x}}$"
        elif self.activation_var.get() == "relu":
            y = np.maximum(0, x)
            derivative = np.where(x > 0, 1, 0)
            formula = r"$f(x) = \max(0, x)$"
        elif self.activation_var.get() == "tanh":
            y = np.tanh(x)
            derivative = 1 - y ** 2
            formula = r"$f(x) = \tanh(x)$"
        elif self.activation_var.get() == "linear":
            y = x
            derivative = np.ones_like(x)
            formula = r"$f(x) = x$"
        self.activation_axes.plot(x, y, 'b-', label='Activation')
        self.activation_axes.plot(x, derivative, 'r--', label='Derivative')
        self.activation_axes.legend()
        self.activation_axes.text(0, -0.5, formula, fontsize=12)
        self.activation_axes.grid(True)
        self.activation_canvas.draw()

    def initialize_network(self):
        self.layer_sizes[1] = self.hidden_neurons_var.get()
        self.activations = [self.activation_var.get()] * (len(self.layer_sizes) - 1)
        self.loss_function = self.loss_var.get()
        self.network = SimpleNeuralNetwork(self.layer_sizes, self.activations, self.loss_function)
        self.current_epoch = 0
        self.epoch_var.set("Epoch: 0")
        self.loss_var_display.set("Loss: N/A")
        self.network.loss_history = []
        self.draw_network()
        self.update_network_visualization()
        self.update_plots()
        self.status_var.set("Initialized")

    def toggle_animation(self):
        if self.is_animating:
            self.is_animating = False
            if self.animation:
                self.animation.event_source.stop()
                self.animation = None
            self.anim_button.config(text="Start Training", bg="#2196f3")
            self.status_var.set("Paused")
        else:
            self.is_animating = True
            self.anim_button.config(text="Stop Training", bg="#f44336")
            self.status_var.set("Training...")
            self.animation = FuncAnimation(self.loss_figure, self.animate, interval=5000 / self.speed_var.get(), blit=False)
            self.loss_canvas.draw()

    def animate(self, frame):
        if not self.is_animating:
            return
        if self.current_epoch < self.max_epochs:
            indices = np.random.permutation(self.X.shape[0])
            X_shuffled = self.X[indices]
            y_shuffled = self.y[indices]
            loss = self.network.backward(X_shuffled[:self.batch_size], y_shuffled[:self.batch_size], self.learning_rate_var.get())
            self.current_epoch += 1
            self.epoch_var.set(f"Epoch: {self.current_epoch}")
            self.loss_var_display.set(f"Loss: {loss:.6f}")
            self.update_network_visualization()
            self.update_plots()
            if self.animation:
                self.animation.event_source.interval = 5000 / self.speed_var.get()
        else:
            self.toggle_animation()
            self.status_var.set("Completed")

    def reset_visualization(self):
        if self.is_animating:
            self.toggle_animation()
        self.initialize_network()
        self.status_var.set("Reset")

if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralNetworkVisualizer1(root)
    root.mainloop()