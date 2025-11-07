import java.util.function.Function;

/**
 * Defines the types of activation functions and their derivatives for use in a {@link Neuron}.
 * <p>
 * This enum encapsulates the logic for an activation function and its corresponding derivative,
 * allowing them to be easily configured and swapped within a neuron. This is crucial for
 * algorithms like backpropagation, which require the derivative to calculate error gradients.
 *
 * <h3>Example Usage</h3>
 * <pre>{@code
 * // Get the sigmoid activation function
 * Function<Double, Double> sigmoid = ActivationType.SIGMOID.getActivation();
 * double activatedValue = sigmoid.apply(0.5); // Result: ~0.622
 *
 * // Get its derivative
 * Function<Double, Double> sigmoidDerivative = ActivationType.SIGMOID.getDerivative();
 * double derivativeValue = sigmoidDerivative.apply(activatedValue); // Result: ~0.235
 * }</pre>
 *
 * @see Neuron
 * @see NeuralNetwork
 * @author Brandon Mejia
 * @version 2025-11-06
 */
public enum ActivationType {

    /**
     * The Sigmoid function.
     * <p>
     * It maps any real value to the (0, 1) interval, making it suitable for models that
     * need to predict probabilities. It is differentiable, which is essential for backpropagation.
     * Its derivative can be efficiently computed from its own output `y` using the formula:
     * `f'(y) = y * (1 - y)`.
     */
    SIGMOID(
        x -> 1.0 / (1.0 + Math.exp(-x)),
        // The derivative is calculated from the function's output 'y' (which is f(x)),
        // not from the original input 'x'. This is a common optimization.
        y -> y * (1.0 - y)
    ),

    /**
     * The Binary Step function.
     * <p>
     * It returns 1 if the input is greater than or equal to 0, and 0 otherwise.
     * This function is not differentiable at x=0 and has a derivative of 0 everywhere else,
     * making it unsuitable for gradient-based training algorithms like backpropagation. It is
     * primarily used for simple threshold-based decisions in non-trainable Perceptrons.
     */
    BINARY_STEP(
        x -> (x >= 0) ? 1.0 : 0.0,
        y -> 0.0 // The derivative is 0 almost everywhere.
    );

    private final Function<Double, Double> activation;
    private final Function<Double, Double> derivative;

    ActivationType(Function<Double, Double> activation, Function<Double, Double> derivative) {
        this.activation = activation;
        this.derivative = derivative;
    }

    /**
     * Returns the activation function.
     * @return A {@code Function<Double, Double>} that takes the net input and returns the activated output.
     */
    public Function<Double, Double> getActivation() {
        return activation;
    }

    /**
     * Returns the derivative of the activation function.
     * @return A {@code Function<Double, Double>} used for backpropagation calculations.
     */
    public Function<Double, Double> getDerivative() {
        return derivative;
    }
}