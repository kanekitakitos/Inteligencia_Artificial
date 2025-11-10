package apps;

import math.Matrix;
import neural.MLP;
import neural.activation.IDifferentiableFunction;
import neural.activation.Step;

import java.util.Scanner;

public class SingleNeuron0101 {

    public static void main(String[] args) {
        // 1. Define the dataset for the logic y = x2 (pattern 0101)
        double[][] inputs = {
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1}
        };
        double[] expectedOutputs = {0, 1, 0, 1};

        // 2. Define the weights and bias calculated manually
        // For y = x2, we can use weights that focus on x2 and ignore x1.
        Matrix weights = new Matrix(new double[][]{{0.0}, {1.0}}); // Shape: 2x1
        double bias = -0.5;

        // 3. Create and configure an MLP to act as a single neuron
        int[] topology = {2, 1}; // 2 inputs, 1 output neuron
        IDifferentiableFunction[] activations = {new Step()};
        MLP mlp = new MLP(topology, activations, -1);

        // Use setters to apply the manually calculated weights and bias
        mlp.setWeights(new Matrix[]{weights});
        mlp.setBiases(new double[]{bias});

        System.out.println("Testing an MLP as a single neuron for the logic y = x2 (pattern 0101):");
        System.out.println("Weights: w1=0.0, w2=1.0, Bias: -0.5");
        System.out.println("-----------------------------------------");

        // Test with the predefined dataset
        Matrix testMatrix = new Matrix(inputs);
        Matrix predictions = mlp.predict(testMatrix);

        for (int i = 0; i < inputs.length; i++) {
            double prediction = predictions.get(i, 0);
            System.out.printf("Input: [%.0f, %.0f] -> Predicted: %.0f, Expected: %.0f\n",
                    inputs[i][0], inputs[i][1], prediction, expectedOutputs[i]);
        }

        // --- Interactive Testing Part ---
        System.out.println("\n--- Interactive Test Mode ---");
        System.out.println("Enter the number of pairs to test, followed by the pairs (e.g., 1.0 0.9)");

        Scanner sc = new Scanner(System.in);
        int n;
        try {
            n = sc.nextInt();
        } catch (java.util.InputMismatchException e) {
            System.err.println("\nError: Invalid input. Please enter an integer for the number of pairs first.");
            sc.close();
            return; // Exit the program gracefully
        }

        for (int i = 0; i < n; i++) {
            // Create an array for the two input values
            double[][] testInput = new double[1][2];
            testInput[0][0] = Double.parseDouble(sc.next().replace(',', '.'));
            testInput[0][1] = Double.parseDouble(sc.next().replace(',', '.'));

            // Get the neuron's prediction
            double prediction = mlp.predict(new Matrix(testInput)).get(0, 0);

            System.out.printf("Input: [%.1f, %.1f] -> Predicted: %d\n", testInput[0][0], testInput[0][1], (int)prediction);
        }
        sc.close();
    }
}