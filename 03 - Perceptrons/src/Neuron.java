import java.util.Random;

/**
 * Represents a single artificial neuron, the fundamental processing unit of a neural network.
 * <p>
 * This class encapsulates the core components of a neuron: a set of input weights, a bias value,
 * and an activation function. Its primary role is to compute a weighted sum of its inputs,
 * add the bias, and then apply the activation function to produce an output.
 * <p>
 * It is designed to be a simple, immutable, and reusable component that can be assembled into layers
 * to form a {@link NeuralNetwork}.
 *
 * @see NeuralNetwork
 * @author Brandon Mejia
 * @version 2025-11-06
 */
public class Neuron {
    private double[] weights;
    private double bias;
    private final ActivationType activationType;
    
    // Fields for backpropagation
    private double output; // Stores the last calculated output
    private double delta;  // Stores the error gradient

    /**
     * Constructs a new Neuron.
     * @param weights The connection weights for the inputs (w1, w2, ...). A defensive copy is made.
     * @param bias The bias value, which is added to the weighted sum (often denoted as w0).
     * @param activationType The type of activation function to use (e.g., SIGMOID).
     */
    public Neuron(double[] weights, double bias, ActivationType activationType) {
        this.weights = weights.clone(); // Use clone to ensure external immutability
        this.bias = bias;
        this.activationType = activationType;
    }

    /**
     * Calculates the neuron's output by performing a forward pass.
     * <p>
     * This method computes the net input (weighted sum of inputs plus bias) and then applies the activation function.
     * @param inputs The input values, typically from the previous layer or the network's main input.
     * @return The activated output of the neuron (e.g., 0 or 1).
     * @throws IllegalArgumentException if the number of inputs does not match the number of weights.
     */
    public double fire(double[] inputs)
    {
        // Safety check: ensure inputs match weights
        if (inputs.length != weights.length) {
            throw new IllegalArgumentException(
                    "Error: Number of inputs (" + inputs.length +
                            ") does not match the number of weights (" + weights.length + ")"
            );
        }

        double sum = bias;
        for (int i = 0; i < weights.length; i++) {
            sum += weights[i] * inputs[i];
        }
        
        // Apply the activation function (e.g., step) to the sum
        this.output = this.activationType.getActivation().apply(sum);
        return this.output;
    }

    /**
     * Calculates the error gradient (delta) for an output layer neuron.
     * @param error The difference between the expected output and the actual output.
     */
    public void calculateDeltaForOutputLayer(double error) {
        this.delta = error * this.activationType.getDerivative().apply(this.output);
    }

    /**
     * Calculates the error gradient (delta) for a hidden layer neuron.
     * @param errorSum The weighted sum of deltas from the next layer.
     */
    public void calculateDeltaForHiddenLayer(double errorSum) {
        this.delta = errorSum * this.activationType.getDerivative().apply(this.output);
    }

    /**
     * Updates the neuron's weights and bias based on the calculated delta and the learning rate.
     * This is the core learning step in backpropagation.
     *
     * @param inputs The input values that were fed into this neuron during the forward pass.
     * @param learningRate The learning rate, which controls the step size of the weight adjustments.
     */
    public void updateWeights(double[] inputs, double learningRate) {
        this.bias += learningRate * delta;
        for (int i = 0; i < weights.length; i++) {
            this.weights[i] += learningRate * delta * inputs[i];
        }
    }

    /**
     * Returns the number of inputs this neuron is configured to accept, which corresponds to the number of weights.
     * @return The count of weights.
     */
    public int getNumberOfInputs() {
        return weights.length;
    }

    /**
     * Returns the bias value (w0) of the neuron.
     * This is required for the {@link NeuralNetwork#visualize()} method.
     * @return The bias value.
     */
    public double getBias() {
        return bias;
    }

    /**
     * Returns the array of input weights (w1, w2, ...).
     * <p>
     * Returns a clone of the internal weights array to protect the neuron's state from external modifications (encapsulation).
     * @return A copy of the weights array.
     */
    public double[] getWeights()
    {
        return this.weights.clone();
    }

    /**
     * Returns the last calculated output of the neuron. Required for backpropagation.
     * @return The last output value.
     */
    public double getOutput() {
        return output;
    }

    /**
     * Returns the calculated error gradient (delta). Required for backpropagation.
     * @return The delta value.
     */
    public double getDelta() {
        return delta;
    }
}