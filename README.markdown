# Neural Network Visualizer

A Python application for visualizing and training a multi-layer neural network using a graphical user interface (GUI) built with Tkinter. The tool allows users to configure network architecture, load datasets (including custom CSV files), train the network, and visualize the network structure, weights, biases, and training loss in real-time.

## Features

- **Interactive GUI**: Configure network parameters such as hidden layers, activation functions, and loss functions.
- **Dataset Support**: Load predefined datasets (e.g., XOR, Iris) or custom datasets via CSV files or manual input.
- **Real-Time Visualization**: Displays the neural network with dynamic updates for weights, biases, and neuron activations.
- **Training Controls**: Train the network with adjustable learning rate and animation speed, step through epochs, or test the trained model.
- **Plots**: Visualizes loss over epochs and the output layer's activation function.
- **CSV Integration**: Automatically populates custom input/output fields with data from loaded CSV files.
- **Error Handling**: Robust validation for dataset formats and network configurations with detailed logging.

## Visualization

Below is a screenshot of the Neural Network Visualizer interface, showing the control panel, network visualization, and training plots:

![Neural Network Visualizer Screenshot](visualization_screenshot.png)

## Requirements

- Python 3.6 or higher
- Libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `tkinter` (included with standard Python installations)

## Installation

1. **Clone the Repository** (or download the code):
   ```bash
   git clone <repository-url>
   cd neural-network-visualizer
   ```

2. **Install Dependencies**:
   ```bash
   pip install numpy pandas matplotlib
   ```

3. **Verify Tkinter**:
   Tkinter is usually included with Python. To confirm, run:
   ```bash
   python -m tkinter
   ```
   If Tkinter is missing, install it:
   - On Ubuntu/Debian: `sudo apt-get install python3-tk`
   - On macOS (with Homebrew): `brew install python-tk`

## Usage

1. **Run the Application**:
   ```bash
   python nn_visualizer.py
   ```
   This launches the GUI with a control panel, network visualization, and plots.

2. **Configure the Network**:
   - **Hidden Layers**: Enter comma-separated integers in the "Hidden Layers" field (e.g., `8,4` for two hidden layers with 8 and 4 neurons).
   - **Activations**: Specify activation functions in the "Activations" field (e.g., `sigmoid,relu,softmax`). Options: `sigmoid`, `relu`, `tanh`, `linear`, `softmax`.
   - **Loss Function**: Select from `RMSE`, `BCE`, or `CCE` in the dropdown menu.
   - Click **Initialize Network** to apply the configuration.

3. **Load a Dataset**:
   - **Predefined Datasets**: Choose a dataset from the dropdown (e.g., `XOR`, `AND`, `Iris`).
   - **Custom Dataset**:
     - Manually enter data in the "Custom Input" and "Custom Output" text fields using Python list syntax (e.g., `[[0,0],[0,1],[1,0],[1,1]]` for input and `[[0],[1],[1],[0]]` for output).
     - Alternatively, load a CSV file:
       - Click **Load CSV** and select a CSV file.
       - In the dialog, check the boxes for feature columns (inputs) and target columns (outputs).
       - Click **Confirm** to load the data.
       - The "Custom Input" and "Custom Output" fields will automatically populate with the loaded data in Python list format.
   - The dataset dropdown will switch to "Custom" after loading a CSV.

4. **Train the Network**:
   - Adjust the **Learning Rate** (0.01 to 0.5) and **Animation Speed** (1 to 100) using the sliders.
   - Click **Start Training** to begin continuous training, or **Step Epoch** to train one epoch at a time.
   - Monitor the loss plot, network visualization (weights, biases, activations), and status bar for progress.
   - Click **Stop Training** to pause the animation.

5. **Test the Network**:
   - Click **Test Network** to display a message box with input data, expected outputs, predicted outputs, and accuracy.

6. **Reset the Network**:
   - Click **Reset** to reinitialize the network and clear training progress.

## Example CSV Format

To load a custom dataset, create a CSV file (e.g., `data.csv`) with numeric data. Example:

```csv
feature1,feature2,target
0.1,0.2,0
0.3,0.4,1
0.5,0.6,0
0.7,0.8,1
```

- **Columns**:
  - `feature1`, `feature2`: Input features (numeric).
  - `target`: Output/target values (numeric).
- **Requirements**:
  - All data must be numeric (integers or floats).
  - No missing or non-numeric values.
  - For `BCE` loss, targets should be approximately 0 or 1 (e.g., -0.1 to 1.1).
  - For `CCE` loss, targets must be one-hot encoded (e.g., `[0,1]` or `[1,0]` for two classes, summing to 1 per row).

After loading, the GUI will display:
- Custom Input: `[[0.1,0.2],[0.3,0.4],[0.5,0.6],[0.7,0.8]]`
- Custom Output: `[[0],[1],[0],[1]]`

## Adding the Visualization Screenshot

To include the visualization screenshot in this README:

1. **Capture the Screenshot**:
   - Run `python nn_visualizer.py`.
   - Take a screenshot of the GUI during operation (e.g., while training or displaying a loaded dataset).
   - **Windows**: Use `Snipping Tool` or `PrtSc` and save as `visualization_screenshot.png`.
   - **macOS**: Press `Cmd + Shift + 4`, select the window, and save.
   - **Linux**: Use `gnome-screenshot` or `ksnapshot`.

2. **Add to Repository**:
   - Place `visualization_screenshot.png` in the project directory.
   - Commit and push:
     ```bash
     git add visualization_screenshot.png
     git commit -m "Add visualization screenshot"
     git push origin main
     ```
   - The README references the image with `![Neural Network Visualizer Screenshot](visualization_screenshot.png)`.

3. **Alternative: Host Externally**:
   - Upload the screenshot to an image hosting service (e.g., Imgur, Postimages).
   - Copy the direct link (e.g., `https://i.imgur.com/XXXXX.png`).
   - Update the README image line:
     ```markdown
     ![Neural Network Visualizer Screenshot](https://i.imgur.com/XXXXX.png)
     ```

## Logging

- The application logs events and errors to `nn_visualizer.log` in the working directory.
- Check this file for debugging if issues occur (e.g., CSV loading errors, invalid configurations).

## Troubleshooting

- **CSV Loading Issues**:
  - Ensure the CSV contains only numeric data with no missing values.
  - Verify that feature and target columns are correctly selected in the dialog.
  - Check `nn_visualizer.log` for detailed error messages.
- **Visualization Problems**:
  - Confirm that `matplotlib` is using the `TkAgg` backend (set in the code).
  - Resize the window if the network display is misaligned.
- **Custom Dataset Errors**:
  - Ensure "Custom Input" and "Custom Output" fields contain valid Python list literals (e.g., `[[0,1],[1,0]]`).
  - Input and output arrays must have compatible shapes (same number of samples).
- **Dependencies**:
  - If libraries are missing, reinstall using `pip install numpy pandas matplotlib`.
  - If Tkinter is unavailable, install it as described in the Installation section.

## Example Workflow

1. Run the application: `python nn_visualizer.py`.
2. Load a CSV file (e.g., `data.csv` as shown above).
3. Select `feature1` and `feature2` as features, `target` as the target in the dialog.
4. Verify that the custom input/output fields populate correctly.
5. Set hidden layers to `8,4`, activations to `sigmoid,sigmoid,sigmoid`, and loss to `BCE`.
6. Click **Initialize Network**, then **Start Training**.
7. Capture a screenshot during training to include in the README.
8. Click **Test Network** to evaluate performance.

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a branch for your feature or bug fix.
3. Submit a pull request with a clear description of changes.

Please report issues via the repository's issue tracker.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details (if included) or refer to the standard MIT License terms.

## Acknowledgments

- Built with Python, Tkinter, NumPy, Pandas, and Matplotlib.
- Inspired by neural network visualization tools for educational purposes.