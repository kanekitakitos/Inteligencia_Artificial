package neural.activation;

import java.util.function.Function;

/**
 * Implements the Hyperbolic Tangent (TanH) activation function.
 * <p>
 * The TanH function squashes its input to a range of [-1, 1], which can help
 * center the data and often leads to faster convergence during training compared
 * to the Sigmoid function.
 * </p>
 * Its derivative is calculated as {@code 1 - fnc(x)^2}.
 *
 * @author Brandon Mejia
 * @version 2025-11-29
 */
public class TanH implements IDifferentiableFunction {

    @Override
    public Function<Double, Double> fnc() { return Math::tanh; }

    @Override
    public Function<Double, Double> derivative() { return y -> 1 - (y * y); }
}