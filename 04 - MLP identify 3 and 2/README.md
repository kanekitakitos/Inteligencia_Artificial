# MLP Digit Classifier (2 vs 3)

A pure Java implementation of a Multi-Layer Perceptron (MLP) neural network designed to classify handwritten digits, specifically distinguishing between the numbers **2** and **3**.

This project was built from scratch without relying on external machine learning libraries (like TensorFlow, PyTorch, or Deeplearning4j). It features a custom matrix mathematics engine, a robust training pipeline, and an automated hyperparameter tuning system.

## 🚀 Key Features

*   **Pure Java Implementation:** Core logic, including matrix operations and backpropagation, is implemented with no external dependencies.
*   **Custom Neural Engine:**
    *   Feedforward and Backpropagation algorithms.
    *   **Momentum** optimization for faster convergence.
    *   **L2 Regularization** (Weight Decay) to prevent overfitting.
    *   Support for **Sigmoid** and **TanH** activation functions.
*   **Automated Hyperparameter Tuning:**
    *   Includes a `HyperparameterTuner` that performs a **Grid Search** to find the best network topology, learning rate, and momentum.
    *   Runs in **parallel** using Java's `ExecutorService` to maximize CPU usage.
    *   Fault-tolerant: Saves progress to `tuning_results.log` and resumes automatically if interrupted.
*   **Data Handling:**
    *   Automatic normalization (Min-Max scaling).
    *   Data splitting (Training vs. Validation).
*   **High Performance:** Achieved **99.00% accuracy** on test datasets with the optimal configuration.

## 📂 Project Structure

The source code is organized into the following packages:

*   **`apps`**: Contains the application logic.
    *   `MLP23.java`: The specific trainer/wrapper for the 2 vs 3 classification task.
    *   `HyperparameterTuner.java`: The parallel grid search utility.
    *   `DataHandler.java`: Utilities for loading CSVs, normalizing pixels, and managing datasets.
*   **`neural`**: The core neural network framework.
    *   `MLP.java`: The generic Multi-Layer Perceptron implementation.
    *   `activation`: Interface and implementations for activation functions (`Sigmoid`, `TanH`).
*   **`math`**:
    *   `Matrix.java`: A custom linear algebra library for matrix multiplication, transposition, and element-wise operations.
*   **`src/models`**: Directory where trained models are serialized and saved.
*   **`tools-for-data`**: A special directory containing scripts designed to analyze and inspect the `.csv` datasets used by the network.

## 🛠️ Setup & Prerequisites

*   **Java:** JDK 17 or higher (required for `record` types used in the tuner).
*   **IDE:** IntelliJ IDEA, Eclipse, or VS Code is recommended.

## 💻 Usage

### 1. Hyperparameter Tuning
To find the best network configuration, run the `HyperparameterTuner`. This will test various combinations of learning rates, momentum, and topologies.

```bash
java apps.HyperparameterTuner
```

*   **Output:** Results are logged to `src/data/tuning_results.log`.
*   **Best Result Found:**
    *   **Topology:** `[400, 2, 1]` (400 inputs, 2 hidden neurons, 1 output)
    *   **LR:** `0.0010`
    *   **Momentum:** `0.99`
    *   **Accuracy:** 99.00%

### 2. Training the Model
You can train a specific model using the `MLP23` class. This handles data loading and the training loop.

```java
// Example usage inside a main method
MLP23 trainer = new MLP23(); // Uses default optimal settings
trainer.train();
MLP model = trainer.getMLP();
model.saveModel("src/models/mlp23_99.model");
```

### 3. Running the Classifier (P4)
The `P4` class is the main entry point for the evaluation/production phase. It loads a pre-trained model and classifies images provided via **Standard Input (stdin)**.

**Input Format:**
*   Each line must contain **400 comma-separated integers** (0-255).
*   These represent the pixel intensity of a 20x20 image.

**Output Format:**
*   The program prints `2` or `3` for each input line.

**How to run:**
```bash
java P4 < input_data.csv
```

## 📊 Performance

Based on the `tuning_results.log`, the network is highly sensitive to Momentum. The best configurations consistently use a momentum of **0.99**.

| Topology | LR | Momentum | Activation | Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| [400, 2, 1] | 0.0010 | 0.99 | Sigmoid, Sigmoid | **99.00%** |
| [400, 4, 1] | 0.0010 | 0.99 | Sigmoid, Sigmoid | 98.88% |
| [400, 3, 1] | 0.0008 | 0.99 | Sigmoid, Sigmoid | 98.50% |

## 👥 Authors

*   **Brandon Mejia**
*   **hdaniel@ualg.pt**

---

*This project was developed for the Artificial Intelligence course (2025).*