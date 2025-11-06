import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Arrays;

/**
 * Represents a feed-forward, multi-layer neural network (MLP).
 * <p>
 * This class orchestrates a collection of {@link Neuron} objects organized into layers.
 * It manages the flow of data from the input layer, through any number of hidden layers,
 * to the output layer. The primary function is to take an initial set of inputs and
 * produce a final prediction.
 *
 * <h3>Building a Network</h3>
 * A network is constructed by providing a list of layers, where each layer is an array of neurons.
 * The structure is defined from the first hidden layer to the output layer.
 *
 * <pre>{@code
 * // Define an activation function
 * Function<Double, Integer> stepFunction = sum -> (sum >= 0) ? 1 : 0;
 *
 * // 1. Define the Hidden Layer
 * Neuron neuronA = new Neuron(new double[]{-2.0, 2.0}, -1.0, stepFunction);
 * Neuron neuronB = new Neuron(new double[]{1.0, 1.0}, -2.0, stepFunction);
 * Neuron[] hiddenLayer = {neuronA, neuronB};
 *
 * // 2. Define the Output Layer
 * Neuron neuronC = new Neuron(new double[]{1.0, 1.0}, -1.0, stepFunction);
 * Neuron[] outputLayer = {neuronC};
 *
 * // 3. Create the list of layers
 * List<Neuron[]> layers = new ArrayList<>();
 * layers.add(hiddenLayer);
 * layers.add(outputLayer);
 *
 * // 4. Instantiate the network
 * NeuralNetwork mlp = new NeuralNetwork(layers);
 *
 * // 5. Make a prediction
 * int prediction = mlp.predict(new double[]{0, 1});
 * }</pre>
 *
 * @see Neuron
 * @author Brandon Mejia
 * @version 2025-11-06
 */
public class NeuralNetwork {

    private final List<Neuron[]> layers;

    /**
     * Constructs a neural network from a list of layers.
     * @param layers A list where each element is a `Neuron[]` array representing a layer.
     *               The list should be ordered from the first hidden layer to the output layer.
     */
    public NeuralNetwork(List<Neuron[]> layers)
    {
        // Cria uma nova lista contendo todos os elementos da lista fornecida.
        // Isto garante que a lista interna da rede seja independente da lista externa.
        this.layers = new ArrayList<>(layers);
    }

    /**
     * Constructs an empty neural network. Layers can be added using the `addLayer` method.
     */
    public NeuralNetwork()
    {
        this.layers = new ArrayList<>();
    }

    /**
     * Adds a new layer to the end of the network.
     * @param neurons A sequence of neurons that will form the new layer.
     */
    public void addLayer(Neuron... neurons) {
        this.layers.add(neurons);
    }

    /**
     * Performs a forward pass through the network to generate a prediction.
     * <p>
     * It processes the input vector through each layer sequentially, with the output of one layer
     * becoming the input for the next.
     *
     * @param inputs The initial input vector for the network. Its size must match the number of inputs
     *               expected by the neurons in the first hidden layer.
     * @return The output of the first neuron in the final layer. This implementation assumes a single-output network.
     */
    public int predict(double[] inputs) 
    {
        double[] currentOutputs = inputs;
        for (Neuron[] layer : layers) {
            double[] nextInputs = new double[layer.length];
            for (int i = 0; i < layer.length; i++) {
                nextInputs[i] = layer[i].fire(currentOutputs);
            }
            currentOutputs = nextInputs;
        }
        return (int) currentOutputs[0];
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
        System.out.println("\n--- Visualização da Arquitetura da Rede Neuronal ---");
        if (layers.isEmpty()) {
            System.out.println("Rede vazia.");
            return;
        }

        // --- FASE 1: Calcular Layout (igual a antes) ---
        int numInputs = layers.get(0)[0].getNumberOfInputs();
        int maxNodes = numInputs;
        for (Neuron[] layer : layers) {
            if (layer.length > maxNodes) {
                maxNodes = layer.length;
            }
        }
        int gridHeight = (maxNodes <= 1) ? 1 : (maxNodes * 2 - 1);
        int numColumns = layers.size() + 1;
        String[][] gridNodes = new String[gridHeight][numColumns];

        // Coluna 0: Entradas
        int numNodes = numInputs;
        int padding = (gridHeight == 1) ? 0 : (gridHeight - (numNodes * 2 - 1)) / 2;
        int rowStep = (gridHeight == 1) ? 1 : 2;
        for (int i = 0; i < numNodes; i++) {
            gridNodes[padding + (i * rowStep)][0] = String.format("[ x%d ]", i + 1);
        }
        // Colunas 1 em diante: Camadas
        for (int i = 0; i < layers.size(); i++) {
            Neuron[] layer = layers.get(i);
            int colIndex = i + 1;
            numNodes = layer.length;
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

        // --- FASE 2: Imprimir Layout (igual a antes) ---
        System.out.println("Layout das Camadas:");
        for (int r = 0; r < gridHeight; r++) {
            StringBuilder line = new StringBuilder();
            for (int c = 0; c < numColumns; c++) {
                String node = gridNodes[r][c] != null ? gridNodes[r][c] : "";
                line.append(String.format("%-10s", node));
                if (c < numColumns - 1) line.append("   ");
            }
            System.out.println(line);
        }

        // --- FASE 3: Imprimir Detalhes (com formato w0, w1, w2...) ---
        System.out.println("\nDetalhes dos Parâmetros e Conexões:");

        String[] inputNames = new String[numInputs];
        for (int k = 0; k < numInputs; k++) {
            inputNames[k] = String.format("[ x%d ]", k + 1);
        }

        for (int i = 0; i < layers.size(); i++) {
            String layerType = (i == layers.size() - 1) ? "Camada de Saída" : "Camada Oculta " + (i + 1);
            String currentPrefix = (i == layers.size() - 1) ? "Out" : "H" + (i + 1);
            System.out.println("  " + layerType + ":");

            String[] prevLayerNames;
            if (i == 0) {
                prevLayerNames = inputNames;
            } else {
                Neuron[] prevLayer = layers.get(i - 1);
                String prevPrefix = "H" + i; // (Corrigido da última vez)
                prevLayerNames = new String[prevLayer.length];
                for (int k = 0; k < prevLayer.length; k++) {
                    prevLayerNames[k] = String.format("[ %s-%d ]", prevPrefix, k + 1);
                }
            }

            // Inspecionar cada neurónio
            for (int j = 0; j < layers.get(i).length; j++) {
                Neuron neuron = layers.get(i)[j];
                String neuronName = String.format("[ %s-%d ]", currentPrefix, j + 1);

                // Imprime o nome do neurónio
                System.out.printf("    - Neurónio %s:%n", neuronName);

                // Imprime o Bias como w0
                System.out.printf("        w0 [Bias] = %.2f%n", neuron.getBias());

                // Imprime os Pesos como w1, w2, ...
                double[] weights = neuron.getWeights();
                for (int k = 0; k < weights.length; k++) {
                    // Mostra TODOS os pesos, incluindo os que são 0.0
                    System.out.printf("        w%d [de %s] = %.2f%n", k + 1, prevLayerNames[k], weights[k]);
                }
            }
        }
        System.out.println("----------------------------------------------------");
    }
}