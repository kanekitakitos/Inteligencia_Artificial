import java.util.function.Function;

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
    private final double[] weights;
    private final double bias;
    private final Function<Double, Integer> activationFunction;

    /**
     * Constructs a new Neuron.
     * @param weights The connection weights for the inputs (w1, w2, ...).
     * @param bias The bias value, which is added to the weighted sum (often denoted as w0).
     * @param activationFunction A function (e.g., a step function) that transforms the net input into the neuron's final output.
     */
    public Neuron(double[] weights, double bias, Function<Double, Integer> activationFunction) {
        this.weights = weights;
        this.bias = bias;
        this.activationFunction = activationFunction;
    }

    /**
     * Calculates the neuron's output by performing a forward pass.
     * <p>
     * This method computes the net input (weighted sum of inputs plus bias) and then applies the activation function.
     * @param inputs The input values, typically from the previous layer or the network's main input.
     * @return The activated output of the neuron (e.g., 0 or 1).
     * @throws IllegalArgumentException if the number of inputs does not match the number of weights.
     */
    public int fire(double[] inputs)
    {
        // Verificação de segurança: garante que os inputs correspondem aos pesos
        if (inputs.length != weights.length) {
            throw new IllegalArgumentException(
                    "Erro: N.º de inputs (" + inputs.length +
                            ") não corresponde ao n.º de pesos (" + weights.length + ")"
            );
        }

        double sum = bias;
        for (int i = 0; i < weights.length; i++) {
            sum += weights[i] * inputs[i];
        }

        // Aplica a função de ativação (ex: degrau) à soma
        return activationFunction.apply(sum);
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
     * This is required for the {@link NeuralNetwork#visualize()} method.
     * @return A copy of the weights array.
     */
    public double[] getWeights() {
        return this.weights;
    }
}