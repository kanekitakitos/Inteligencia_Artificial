package apps;

import math.Matrix;
import neural.MLP;
import neural.activation.IDifferentiableFunction;
import neural.activation.Sigmoid;
import neural.activation.TanH;

/**
 * A factory and trainer for a specific Multi-Layer Perceptron (MLP) model designed to classify handwritten digits '2' and '3'. This class encapsulates a pre-defined MLP configuration and acts as a high-level trainer.
 * <p>
 * It simplifies the process of preparing and training the model by orchestrating the {@link DataHandler} for data loading and the {@link MLP} for the core training execution. The primary training logic, which includes asynchronous validation and best-model checkpointing, is handled by the {@link MLP#train(Matrix, Matrix, Matrix, Matrix, double, int, double)} method. This class serves as a convenient wrapper to instantiate and run that process with a proven configuration.
 * </p>
 *
 * <h3>Example Usage</h3>
 * <p>
 * This class is designed to be used programmatically as a factory for a trained model. The following example demonstrates how to use {@code MLP23} to get a trained model and use it for predictions.
 * </p>
 * <ul>
 *   <li><b>Implementation:</b>
 *      <pre>{@code
 * // 1. Instantiate the trainer factory.
 * MLP23 trainer = new MLP23();
 *
 * // 2. Execute the training process. This loads and processes data automatically.
 * trainer.train();
 *
 * // 3. Retrieve the best-performing model after training.
 * MLP bestModel = trainer.getMLP();
 *
 * // 4. Use the model to make a prediction on a new data sample.
 * // (Assuming 'newImageMatrix' is a 1x400 Matrix).
 * Matrix prediction = bestModel.predict(newImageMatrix);
 * long label = Math.round(prediction.get(0, 0));
 * System.out.println(label == 0 ? "Predicted: 2" : "Predicted: 3");
 *      }</pre>
 *   </li>
 * </ul>
 *
 * <pre>{@code
 * // 1. Instantiate the trainer factory.
 * MLP23 trainer = new MLP23();
 *
 * // 2. Execute the training process. This loads and processes data automatically.
 * trainer.train();
 *
 * // 3. Retrieve the best-performing model after training.
 * MLP bestModel = trainer.getMLP();
 *
 * // 4. Use the model to make a prediction on a new data sample.
 * // (Assuming 'newImageMatrix' is a 1x400 Matrix).
 * Matrix prediction = bestModel.predict(newImageMatrix);
 * long label = Math.round(prediction.get(0, 0));
 * System.out.println(label == 0 ? "Predicted: 2" : "Predicted: 3");
 * }</pre>
 *
 * @see MLP
 * @see DataHandler
 * @see IDifferentiableFunction
 * @see HyperparameterTuner
 *
 * @author Brandon Mejia
 * @version 2025-12-02
 */
public class MLP23
{

    private final double lr;
    private final int epochs;
    private final double momentum;
    private final MLP mlp;
    public static final int SEED = 8; // 2;4;5 5:00 ;7;8 4:21 ;16 4:17


    /**
     * Constructs the MLP trainer with a predefined network topology and activation functions.
     */
    public MLP23()
    {
        this(new int[]{400, 4, 1}, new IDifferentiableFunction[]{new TanH(), new Sigmoid()}, 0.01, 0.9, 30000);
    }

    /**
     * Constructs the MLP trainer with a specific network configuration and hyperparameters.
     * This constructor is ideal for hyperparameter tuning.
     *
     * @param topology The network layer sizes.
     * @param functions The activation functions for each layer transition.
     * @param lr The learning rate.
     * @param momentum The momentum factor.
     * @param epochs The number of training epochs.
     */
    public MLP23(int[] topology, IDifferentiableFunction[] functions, double lr, double momentum, int epochs)
    {
        if (topology.length - 1 != functions.length) {
            throw new IllegalArgumentException("The number of activation functions must be one less than the number of layers.");
        }
        this.lr = lr;
        this.momentum = momentum;
        this.epochs = epochs;
        this.mlp = new MLP(topology, functions, SEED);
    }

    /**
     * A high-level training method that orchestrates the entire process from file paths.
     * <p>
     * This method serves as a convenient entry point for training. It uses the {@link DataHandler}
     * to load, preprocess, and split the data into training and validation sets before
     * initiating the core training loop.
     * </p>
     */
    public void train()
    {
        // 1. Load and prepare the data using DataHandler
        DataHandler dataManager = new DataHandler(SEED, DataHandler.NormalizationType.MIN_MAX);
        // 2. Call the core training method with the prepared matrices
        train(dataManager.getTrainInputs(), dataManager.getTrainOutputs(), dataManager.getTestInputs(), dataManager.getTestOutputs());
    }

    /**
     * Trains the MLP model using the provided training and validation datasets.
     *
     * @param trainInputs  The matrix of training input data.
     * @param trainOutputs The matrix of training label data.
     * @param valInputs    The matrix of validation input data.
     * @param valOutputs   The matrix of validation label data.
     * @return The best validation error (MSE) achieved during training.
     */
    public double train(Matrix trainInputs, Matrix trainOutputs, Matrix valInputs, Matrix valOutputs)
    {
        // The complex training logic is now encapsulated within the MLP class itself.
        return this.mlp.train(trainInputs, trainOutputs, valInputs, valOutputs, this.lr, this.epochs, this.momentum);
    }

    /**
     * A simple data class to hold test evaluation metrics.
     */
    public record TestMetrics(double accuracy, double precision, double recall, double f1Score) {}

    /**
     * Evaluates the trained model against a test dataset and computes performance metrics.
     *
     * @param testInputs The test input data.
     * @param testOutputs The test target labels.
     * @return A {@link TestMetrics} object containing accuracy, precision, recall and F1-score.
     */
    public TestMetrics test(Matrix testInputs, Matrix testOutputs) {
        int truePositives = 0;
        int falsePositives = 0;
        int falseNegatives = 0;
        int correctPredictions = 0;

        for (int i = 0; i < testOutputs.rows(); i++) {
            // Extrai uma linha de cada vez para a predição
            double[] row = new double[testInputs.cols()];
            for (int j = 0; j < testInputs.cols(); j++) {
                row[j] = testInputs.get(i, j);
            }
            Matrix inputRow = new Matrix(new double[][]{row});

            Matrix prediction = mlp.predict(inputRow);
            long predictedLabel = Math.round(prediction.get(0, 0));
            double actualValue = testOutputs.get(i, 0);

            if (predictedLabel == actualValue) {
                correctPredictions++;
            }

            // Calcula TP, FP, FN (considerando '1' como a classe positiva)
            if (predictedLabel == 1 && actualValue == 1) {
                truePositives++;
            } else if (predictedLabel == 1 && actualValue == 0) {
                falsePositives++;
            } else if (predictedLabel == 0 && actualValue == 1) {
                falseNegatives++;
            }
        }

        double accuracy = (double) correctPredictions / testOutputs.rows() * 100.0;

        double precision = (truePositives + falsePositives > 0) ? (double) truePositives / (truePositives + falsePositives) : 0.0;
        double recall = (truePositives + falseNegatives > 0) ? (double) truePositives / (truePositives + falseNegatives) : 0.0;
        double f1Score = (precision + recall > 0) ? 2 * (precision * recall) / (precision + recall) : 0.0;

        return new TestMetrics(accuracy, precision, recall, f1Score);
    }

    /**
     * Retrieves the fully trained Multi-Layer Perceptron model.
     * <p>This is the best-performing model found during the training process, selected based on the lowest validation error.</p>
     * @return The trained {@link MLP} instance.
     */
    public MLP getMLP() { return this.mlp; }
}