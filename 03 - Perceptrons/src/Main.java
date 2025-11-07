import java.util.Random;

public class Main {
    public static void main(String[] args) {
        // Define the dataset for the logic y = x2
        double[][] inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

        double[][] expected = { {0},
                                {1},
                                {0},
                                {1} };

        //demonstrateManualMLP(inputs, expected);
        demonstrateTrainableMLP(inputs, expected);
    }

    /**
     * Demonstrates an MLP with manually defined weights and structure to solve the y=x2 logic.
     * Uses the BINARY_STEP activation function.
     */
    private static void demonstrateManualMLP(double[][] inputs, double[][] expected) {
        System.out.println("\n--- Demonstration 1: MLP with Manual Weights (Step Function) ---");
        System.out.println("Logic: y = (¬x1 AND x2) OR (x1 AND x2), which is equivalent to y = x2");

        // Manual configuration of neurons to replicate the boolean logic
        Neuron neuronA = new Neuron(new double[]{-2.0, 2.0}, -1.0, ActivationType.BINARY_STEP); // ¬x1 AND x2
        Neuron neuronB = new Neuron(new double[]{1.0, 1.0}, -2.0, ActivationType.BINARY_STEP); //  x1 AND x2
        Neuron neuronC = new Neuron(new double[]{1.0, 1.0}, -1.0, ActivationType.BINARY_STEP); //     A OR B

        // Network construction
        NeuralNetwork manualMlp = new NeuralNetwork();
        manualMlp.addLayer(neuronA, neuronB); // Hidden Layer
        manualMlp.addLayer(neuronC);          // Output Layer

        // Teste e visualização
        System.out.println("\nResultados da Rede Manual:");
        printResultsHeader("MLP Manual");
        for (int i = 0; i < inputs.length; i++) {
            double[] prediction = manualMlp.predict(inputs[i]);
            int result = (int) prediction[0];
            printResultRow(inputs[i], (int) expected[i][0], result);
        }
        manualMlp.visualize();
    }

    /**
     * Demonstrates an MLP that learns the y=x2 logic through the backpropagation algorithm.
     * Uses the SIGMOID activation function.
     */
    private static void demonstrateTrainableMLP(double[][] inputs, double[][] expected) {
        System.out.println("\n--- Demonstration 2: MLP with Training via Backpropagation ---");

        // Network architecture construction
        Random random = new Random(4); // Seed for reproducible results
        NeuralNetwork mlp = new NeuralNetwork();
        mlp.addLayer(2, 2, ActivationType.SIGMOID, random); // Hidden Layer
        mlp.addLayer(1, 2, ActivationType.SIGMOID, random); // Output Layer

//        System.out.println("\nArquitetura da Rede ANTES do Treino:");
//        mlp.visualize();

        // Network training
        System.out.println("\nStarting training...");
        int epochs = 10000;
        double learningRate = 0.1;
        mlp.train(inputs, expected, epochs, learningRate);
        System.out.printf("Training complete.%n");

        System.out.println("\nNetwork Architecture AFTER Training:");
        mlp.visualize();

        // Teste da rede treinada
        System.out.println("\nResultados da Rede Treinada:");
        printResultsHeader("MLP Treinada");
        for (int i = 0; i < inputs.length; i++) {
            double[] prediction = mlp.predict(inputs[i]);
            double probability = prediction[0];
            int result = (probability >= 0.5) ? 1 : 0;
            printResultRow(inputs[i], (int) expected[i][0], result, probability);
        }
    }

    private static void printResultsHeader(String modelName) {
        // Adjust header if the model is probabilistic
        if (modelName.contains("Trained")) {
            System.out.println("x1 | x2 | Expected | " + modelName + " Result | Probability");
            System.out.println("---|----|----------|------------------|-------------");
        } else {
            System.out.println("x1 | x2 | Expected | " + modelName + " Result");
            System.out.println("---|----|----------|------------------");
        }
    }

    private static void printResultRow(double[] input, int expected, int result, double probability) {
        System.out.printf("%2.0f | %2.0f | %-8d | %-16d | %.4f%n", input[0], input[1], expected, result, probability);
    }

    // Overload for non-probabilistic results
    private static void printResultRow(double[] input, int expected, int result) {
        System.out.printf("%2.0f | %2.0f | %-8d | %d%n", input[0], input[1], expected, result);
    }
}
