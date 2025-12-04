package neural.activation;

import java.util.function.Function;

/**
 * @author hdaniel@ualg.pt
 * @version 202511051244
 */
public class Step implements IDifferentiableFunction {

    static private double threshold = 0.5;


    public static void setThreshold(double threshold) {
        Step.threshold = threshold;
    }

    @Override
    public Function<Double, Double> fnc() {

        // insert code here to return a lambda function that returns
        // 0 if its argument is < 0 and 1 otherwise
        return x -> x < threshold ? 0.0 : 1.0;
    }

    @Override
    public Function<Double, Double> derivative() {
        throw new UnsupportedOperationException("Step function is not differentiable.");
    }

}
