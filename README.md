# Visualization-of-Neural-Network

## Overview
This project is an interactive Neural Network Visualizer built using Python, Tkinter, and Matplotlib. It provides a graphical interface to design, train, and visualize multi-layer neural networks. Users can experiment with different network architectures, activation functions, loss functions, and datasets, observing real-time updates of weights, biases, and loss during training. The visualizer is designed for educational purposes, helping users understand the inner workings of neural networks through dynamic visualizations.

## Features
- **Interactive GUI**: Configure network parameters such as hidden layers, activation functions, and learning rate via a control panel.
- **Real-Time Visualization**: Displays neurons, connections, weights, biases, and activation values during training.
- **Dataset Support**: Includes predefined datasets (XOR, AND, OR, NAND, NOR, XNOR, Circles, Moons, Iris) and supports custom datasets via CSV or manual input.
- **Loss and Activation Plots**: Visualizes loss over epochs and the output layer's activation function with its derivative.
- **Training Controls**: Options to start/stop training, step through epochs, reset the network, and test predictions.
- **Error Handling**: Robust logging and user-friendly error messages for invalid inputs or configurations.
- **Customizable**: Supports various activation functions (sigmoid, relu, tanh, linear, softmax) and loss functions (RMSE, BCE, CCE).

## Requirements
To run the Neural Network Visualizer, ensure you have the following Python libraries installed:
- `tkinter` (usually included with Python)
- `numpy`
- `pandas`
- `matplotlib`

Install the dependencies using:
```bash
pip install numpy pandas matplotlib
