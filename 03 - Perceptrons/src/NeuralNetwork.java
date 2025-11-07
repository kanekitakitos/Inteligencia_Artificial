import java.util.ArrayList;
import java.util.List;
import java.util.Random;
/**
 * Represents a feed-forward, multi-layer neural network (MLP).
 * <p>
 * This class orchestrates a collection of {@link Neuron} objects organized into layers.
 * It manages the flow of data from the input layer, through any number of hidden layers,
 * to the output layer. The primary function is to take an initial set of inputs and
 * produce a final prediction. It also implements the backpropagation algorithm to train
 * the network's weights and biases.
 *
 * <h3>Example Usage</h3>
 * A network can be built layer by layer, either by adding pre-configured neurons
 * or by specifying the layer's architecture and letting the network initialize them.
 *
 * <h4>1. Manual Network Construction</h4>
 * <p>Ideal for simple, non-trainable models or for replicating known logic.</p>
 * <pre>{@code
 * // 1. Create the network instance
 * NeuralNetwork mlp = new NeuralNetwork();
 *
 * // 2. Define and add the hidden layer
 * Neuron neuronA = new Neuron(new double[]{-2.0, 2.0}, -1.0, ActivationType.BINARY_STEP);
 * Neuron neuronB = new Neuron(new double[]{1.0, 1.0}, -2.0, ActivationType.BINARY_STEP);
 * mlp.addLayer(neuronA, neuronB);
 *
 * // 3. Define and add the output layer
 * Neuron neuronC = new Neuron(new double[]{1.0, 1.0}, -1.0, ActivationType.BINARY_STEP);
 * mlp.addLayer(neuronC);
 *
 * // 4. Make a prediction
 * double[] prediction = mlp.predict(new double[]{0, 1}); // Returns {1.0}
 * }</pre>
 *
 * <h4>2. Trainable Network Construction</h4>
 * <p>The common approach for machine learning, where the network learns from data.</p>
 * <pre>{@code
 * // 1. Create the network and a random number generator for weights
 * NeuralNetwork mlp = new NeuralNetwork();
 * Random random = new Random();
 *
 * // 2. Add layers by specifying their architecture
 * // Hidden layer: 2 neurons, each taking 2 inputs
 * mlp.addLayer(2, 2, ActivationType.SIGMOID, random);
 * // Output layer: 1 neuron, taking 2 inputs (from the hidden layer)
 * mlp.addLayer(1, 2, ActivationType.SIGMOID, random);
 *
 * // 3. Train the network
 * mlp.train(inputs, expectedOutputs, 10000, 0.1);
 *
 * }</pre>
 *
 * @see Neuron
 * @author Brandon Mejia
 * @version 2025-11-06
 */
public class NeuralNetwork {

    private final List<Layer> layers;

    /**
     * Constructs an empty neural network. Layers can be added using the `addLayer` method.
     */
    public NeuralNetwork() {
        this.layers = new ArrayList<>();
    }
    
    /**
     * Adds a new layer of trainable neurons, initializing them with random weights and biases.
     *
     * @param numberOfNeurons The number of neurons in this layer.
     * @param numberOfInputsPerNeuron The number of inputs for each neuron in this layer.
     * @param activationType The activation function to be used by all neurons in this layer.
     * @param random A Random instance for weight and bias initialization.
     */
    public void addLayer(int numberOfNeurons, int numberOfInputsPerNeuron, ActivationType activationType, Random random) {
        this.layers.add(new Layer(numberOfNeurons, numberOfInputsPerNeuron, activationType, random));
    }

    /**
     * Adds a pre-configured layer of neurons to the network.
     * @param neurons A sequence of manually constructed neurons that will form the new layer.
     */
    public void addLayer(Neuron... neurons) {
        this.layers.add(new Layer(neurons));
    }

    /**
     * Performs a forward pass through the network to generate a prediction.
     * <p>
     * It processes the input vector through each layer sequentially, with the output of one layer
     * becoming the input for the next.
     *
     * @param inputs The initial input vector for the network. Its size must match the number of inputs
     *               expected by the neurons in the first hidden layer.
     * @return An array containing the outputs from all neurons in the final layer.
     */
    public double[] predict(double[] inputs)
    {
        double[] currentOutputs = inputs;
        for (Layer layer : layers) {
            Neuron[] neurons = layer.getNeurons();
            double[] nextInputs = new double[neurons.length];
            for (int i = 0; i < neurons.length; i++) {

                nextInputs[i] = neurons[i].fire(currentOutputs);
            }
            currentOutputs = nextInputs;
        }
        return currentOutputs;
    }

    /**
     * Trains the neural network using the backpropagation algorithm for a fixed number of epochs.
     *
     * @param trainingInputs Array of input vectors for training.
     * @param trainingOutputs Array of expected output vectors.
     * @param epochs The number of iterations over the entire dataset.
     * @param learningRate The learning rate for adjusting weights and biases.
     */
    public void train(double[][] trainingInputs, double[][] trainingOutputs, int epochs, double learningRate) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double epochError = runTrainingEpoch(trainingInputs, trainingOutputs, learningRate);

            // Optional: Print the average error every N epochs to track progress
//            if ((epoch + 1) % 1000 == 0 || epoch == 0)
//                System.out.printf("Epoch %d/%d, Average Error: %.6f%n", epoch + 1, epochs, epochError);

        }
    }

    /**
     * Treina a rede neural até que um erro alvo seja alcançado ou um número máximo de épocas seja atingido.
     *
     * @param trainingInputs Array of input vectors for training.
     * @param trainingOutputs Array of expected output vectors.
     * @param learningRate The learning rate.
     * @param targetError The target mean squared error to stop training.
     * @param maxEpochs The maximum number of epochs to prevent an infinite loop.
     */
    public void train(double[][] trainingInputs, double[][] trainingOutputs, double learningRate, double targetError, int maxEpochs) {
        double averageError = Double.MAX_VALUE;
        for (int epoch = 0; epoch < maxEpochs && averageError > targetError; epoch++) {
            averageError = runTrainingEpoch(trainingInputs, trainingOutputs, learningRate);

            if ((epoch + 1) % 1000 == 0 || epoch == 0) {
                System.out.printf("Epoch %d/%d, Average Error: %.6f%n", epoch + 1, maxEpochs, averageError);
            }
        }
        System.out.printf("Training finished with Average Error: %.6f%n",
                // The epoch count here is complex to determine if stopped by targetError, so it's omitted.
                averageError);
    }

    private double runTrainingEpoch(double[][] trainingInputs, double[][] trainingOutputs, double learningRate) {
        double totalError = 0;
        for (int i = 0; i < trainingInputs.length; i++) {
            double[] inputs = trainingInputs[i];
            double[] expected = trainingOutputs[i];

            // 1. Forward Pass (executed by predict)
            double[] actual = predict(inputs);

            // 2. Backward Pass (Error and Delta Calculation)
            Neuron[] outputLayer = layers.get(layers.size() - 1).getNeurons();
            for (int j = 0; j < outputLayer.length; j++) {
                double error = expected[j] - actual[j];
                outputLayer[j].calculateDeltaForOutputLayer(error);
                totalError += Math.pow(error, 2);
            }

            for (int j = layers.size() - 2; j >= 0; j--) {
                Neuron[] hiddenLayer = layers.get(j).getNeurons();
                Neuron[] nextLayer = layers.get(j + 1).getNeurons();
                for (int k = 0; k < hiddenLayer.length; k++) {
                    double errorSum = 0;
                    for (int l = 0; l < nextLayer.length; l++) {
                        errorSum += nextLayer[l].getWeights()[k] * nextLayer[l].getDelta();
                    }
                    hiddenLayer[k].calculateDeltaForHiddenLayer(errorSum);
                }
            }

            // 3. Update Weights and Biases
            for (int j = 0; j < layers.size(); j++) {
                Layer currentLayer = layers.get(j);
                // The inputs for the current layer are the outputs of the previous layer.
                // For the first layer (j=0), the inputs are from the training set.
                double[] inputsForLayer = (j == 0) ? inputs : getLayerOutputs(layers.get(j - 1));

                for (Neuron neuron : currentLayer.getNeurons()) {
                    neuron.updateWeights(inputsForLayer, learningRate);
                }
            }
        }
        return totalError / trainingInputs.length;
    }

    /**
     * Prints a detailed text-based visualization of the network's architecture.
     * <p>
     * This method displays the network's structure, including the layout of its layers and nodes,
     * followed by a comprehensive list of all parameters (bias and weights) for each neuron.
     * This is useful for debugging and verifying the network's configuration.
     */
    public void visualize() 
    {
        System.out.println("\n--- Neural Network Architecture Visualization ---");
        if (layers.isEmpty()) {
            System.out.println("The network is empty.");
            return;
        }

        // --- PHASE 1: Calculate Layout ---
        int numInputs = layers.get(0).getNeurons()[0].getNumberOfInputs();
        int maxNodes = numInputs;
        for (Layer layer : layers) {
            if (layer.getNeurons().length > maxNodes) {
                maxNodes = layer.getNeurons().length;
            }
        }
        int gridHeight = (maxNodes <= 1) ? 1 : (maxNodes * 2 - 1);
        int numColumns = layers.size() + 1;
        String[][] gridNodes = new String[gridHeight][numColumns];

        // Column 0: Inputs
        int numNodes = numInputs;
        int padding = (gridHeight == 1) ? 0 : (gridHeight - (numNodes * 2 - 1)) / 2;
        int rowStep = (gridHeight == 1) ? 1 : 2;
        for (int i = 0; i < numNodes; i++) {
            gridNodes[padding + (i * rowStep)][0] = String.format("[ x%d ]", i + 1);
        }
        // Columns 1 onwards: Layers
        for (int i = 0; i < layers.size(); i++) {
            Neuron[] neurons = layers.get(i).getNeurons();
            int colIndex = i + 1;
            numNodes = neurons.length;
            padding = (gridHeight == 1) ? 0 : (gridHeight - (numNodes * 2 - 1)) / 2;
            rowStep = (gridHeight == 1) ? 1 : 2;
            if (numNodes == 1 && gridHeight > 1) {
                padding = gridHeight / 2;
                rowStep = 1;
            }
            for (int j = 0; j < numNodes; j++) {
                String layerPrefix = (i == layers.size() - 1) ? "Out" : "H" + (i + 1);
                gridNodes[padding + (j * rowStep)][colIndex] = String.format("[ %s-%d ]", layerPrefix, j + 1);
            }
        }

        // --- PHASE 2: Print Layout ---
        System.out.println("Layer Layout:");
        for (int r = 0; r < gridHeight; r++) {
            StringBuilder line = new StringBuilder();
            for (int c = 0; c < numColumns; c++) {
                String node = gridNodes[r][c] != null ? gridNodes[r][c] : "";
                line.append(String.format("%-10s", node));
                if (c < numColumns - 1) line.append("   ");
            }
            System.out.println(line);
        }

        // --- PHASE 3: Print Parameter and Connection Details ---
        System.out.println("\nParameter and Connection Details:");

        String[] inputNames = new String[numInputs];
        for (int k = 0; k < numInputs; k++) {
            inputNames[k] = String.format("[ x%d ]", k + 1);
        }

        for (int i = 0; i < layers.size(); i++) {
            String layerType = (i == layers.size() - 1) ? "Output Layer" : "Hidden Layer " + (i + 1);
            String currentPrefix = (i == layers.size() - 1) ? "Out" : "H" + (i + 1);
            System.out.println("  " + layerType);

            String[] prevLayerNames;
            if (i == 0) {
                prevLayerNames = inputNames;
            } else {
                Neuron[] prevLayer = layers.get(i - 1).getNeurons();
                String prevPrefix = "H" + i;
                prevLayerNames = new String[prevLayer.length];
                for (int k = 0; k < prevLayer.length; k++) {
                    prevLayerNames[k] = String.format("[ %s-%d ]", prevPrefix, k + 1);
                }
            }

            // Inspecionar cada neurónio
            System.out.println("  ------------------------------------");
            for (int j = 0; j < layers.get(i).getNeurons().length; j++) {
                Neuron neuron = layers.get(i).getNeurons()[j];
                String neuronName = String.format("[ %s-%d ]", currentPrefix, j + 1);

                // Print neuron name
                System.out.printf("    - Neuron %s:%n", neuronName);

                // Print Bias as w0
                System.out.printf("        w0 [Bias] = %.2f%n", neuron.getBias());

                // Print Weights as w1, w2, ...
                double[] weights = neuron.getWeights();
                for (int k = 0; k < weights.length; k++) {
                    // Show ALL weights, including those that are 0.0
                    System.out.printf("        w%d [from %s] = %.2f%n", k + 1, prevLayerNames[k], weights[k]);
                }
            }
        }
    }

    /**
     * Helper method to extract the outputs from all neurons in a given layer.
     * @param layer The layer from which to extract the outputs.
     * @return A double array containing the outputs of the neurons.
     */
    private double[] getLayerOutputs(Layer layer) {
        Neuron[] neurons = layer.getNeurons();
        double[] outputs = new double[neurons.length];
        for (int i = 0; i < neurons.length; i++) {
            outputs[i] = neurons[i].getOutput();
        }
        return outputs;
    }




    // -------------------------------------------------------------------------------------------------------------------------

    /**
     * Represents a single layer in a neural network.
     * <p>
     * A layer is a collection of neurons that operate together. This class encapsulates
     * the creation and management of neurons for a specific layer, ensuring they all
     * share the same activation function and are initialized correctly.
     * It is defined as a private static nested class because it is an
     * implementation detail of the NeuralNetwork.
     *
     * @see Neuron
     */
    private static class Layer
    {
        private final Neuron[] neurons;

        /**
         * Constructs a layer by specifying its properties. Neurons are created automatically
         * with random weights and biases, ready for training.
         */
        public Layer(int numberOfNeurons, int numberOfInputsPerNeuron, ActivationType activationType, Random random) {
            this.neurons = new Neuron[numberOfNeurons];
            for (int i = 0; i < numberOfNeurons; i++) {
                // The logic for creating random weights and biases belongs to the Layer.
                double[] weights = new double[numberOfInputsPerNeuron];
                for (int j = 0; j < numberOfInputsPerNeuron; j++) {
                    weights[j] = random.nextDouble() * 2 - 1; // Random weights between -1 and 1
                }
                double bias = random.nextDouble() * 2 - 1; // Random bias between -1 and 1

                this.neurons[i] = new Neuron(weights, bias, activationType);
            }
        }

        /**
         * Constructs a layer from a pre-configured set of neurons.
         */
        public Layer(Neuron... neurons) {
            this.neurons = neurons;
        }

        /**
         * @return The array of neurons that make up this layer.
         */
        public Neuron[] getNeurons() {
            return neurons;
        }
    }

    // -------------------------------------------------------------------------------------------------------------------------

}