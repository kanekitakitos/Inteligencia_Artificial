package neural.activation;

import java.util.function.Function;

/**
 * Implements the Leaky Rectified Linear Unit (Leaky ReLU) activation function.
 * <p>
 * Leaky ReLU is a variant of the ReLU function that allows a small, non-zero gradient
 * when the unit is not active (i.e., for negative inputs). This helps to mitigate the
 * "dying ReLU" problem, where neurons can become stuck during training if they consistently
 * output zero.
 * </p>
 * <p>
 * The function is defined as:
 * <ul>
 *   <li>{@code f(x) = x} if {@code x > 0}</li>
 *   <li>{@code f(x) = alpha * x} if {@code x <= 0}</li>
 * </ul>
 * The {@code alpha} parameter is a small constant, typically 0.01.
 * </p>
 *
 * <h3>Example Usage</h3>
 * <pre>{@code
 * // Using Leaky ReLU with the default alpha (0.01)
 * IDifferentiableFunction leakyReLUDefault = new LeakyReLU();
 *
 * // Using Leaky ReLU with a custom alpha
 * IDifferentiableFunction leakyReLUCustom = new LeakyReLU(0.05);
 *
 * // In an MLP
 * int[] topology = {784, 128, 10};
 * IDifferentiableFunction[] functions = {new LeakyReLU(), new Sigmoid()};
 * MLP mlp = new MLP(topology, functions, 42);
 * }</pre>
 *
 * @see ReLU
 * @see Sigmoid
 * @see TanH
 * @author Gemini Code Assist
 * @version 2025-11-29
 */
public class LeakyReLU implements IDifferentiableFunction {

    private final double alpha;

    /**
     * Constructs a LeakyReLU activation function with the default alpha value of 0.01.
     */
    public LeakyReLU() {
        this(0.01);
    }

    /**
     * Constructs a LeakyReLU activation function with a custom alpha value.
     * @param alpha The coefficient for negative inputs.
     */
    public LeakyReLU(double alpha) {
        this.alpha = alpha;
    }

    @Override
    public Function<Double, Double> fnc() {
        return x -> (x > 0) ? x : alpha * x;
    }

    @Override
    public Function<Double, Double> derivative() {
        return x -> (x > 0) ? 1.0 : alpha;
    }
}