package test;

import math.Matrix;
import neural.MLP;
import neural.activation.IDifferentiableFunction;
import neural.activation.Sigmoid;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MLPTest {

    @Test
    @DisplayName("Predict should produce an output matrix with correct dimensions")
    void testPredictDimensions() {
        int[] topology = {2, 3, 1};
        IDifferentiableFunction[] activations = {new Sigmoid(), new Sigmoid()};
        MLP mlp = new MLP(topology, activations, 42);

        Matrix input = new Matrix(new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}});
        Matrix output = mlp.predict(input);

        assertEquals(4, output.rows(), "Output rows should match input rows");
        assertEquals(1, output.cols(), "Output columns should match output layer size");
    }

    @Test
    @DisplayName("Training should reduce the Mean Squared Error over epochs for a simple problem (AND gate)")
    void testTrainReducesError() {
        int[] topology = {2, 2, 1}; // 2 inputs, 2 hidden neurons, 1 output
        IDifferentiableFunction[] activations = {new Sigmoid(), new Sigmoid()};
        MLP mlp = new MLP(topology, activations, 12345);

        // Dataset for AND gate
        Matrix trX = new Matrix(new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}});
        Matrix trY = new Matrix(new double[][]{{0}, {0}, {0}, {1}});

        int epochs = 5000;
        double learningRate = 0.5;

        double[] mseHistory = mlp.train(trX, trY, learningRate, epochs);

        assertNotNull(mseHistory);
        assertEquals(epochs, mseHistory.length);

        // The error at the beginning should be higher than the error at the end
        double initialError = mseHistory[0];
        double finalError = mseHistory[epochs - 1];

        System.out.printf("Initial MSE: %.5f, Final MSE: %.5f%n", initialError, finalError);

        assertTrue(finalError < initialError, "Final error should be less than initial error");
        assertTrue(finalError < 0.01, "Final error should be very small after training");
    }
}
