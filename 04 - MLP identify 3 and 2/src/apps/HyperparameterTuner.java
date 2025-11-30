package apps;

import neural.MLP;
import neural.activation.IDifferentiableFunction;
import neural.activation.Sigmoid;
import neural.activation.TanH;
import neural.activation.ReLU;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;

/**
 * Orchestrates an automated search for the optimal hyperparameters for an MLP model.
 * <p>
 * This class implements a parallelized Grid Search strategy to systematically explore
 * multiple combinations of hyperparameters, such as learning rate, momentum, network topology,
 * and activation functions. By leveraging a thread pool, it can evaluate several model
 * configurations concurrently, significantly reducing the time required to find a
 * high-performing setup.
 * </p>
 * <p>
 * The results of each trial are collected, sorted by validation error, and presented
 * in a summary report, highlighting the best combination found.
 * </p>
 *
 * <h3>Example Usage</h3>
 * <p>
 * To start the hyperparameter tuning process, simply create an instance of this class
 * and invoke the {@link #runGridSearch()} method.
 * </p>
 * <pre>{@code
 * public class TuneModel {
 *     public static void main(String[] args) {
 *         // 1. Instantiate the tuner.
 *         HyperparameterTuner tuner = new HyperparameterTuner();
 *
 *         // 2. Execute the grid search.
 *         // This will train and evaluate models for all defined hyperparameter combinations.
 *         tuner.runGridSearch();
 *
 *         // 3. The results, including the best parameter set, will be printed to the console.
 *     }
 * }
 * }</pre>
 *
 * <h3>Further Improvements</h3>
 * <p>
 * While Grid Search is exhaustive, it can be computationally expensive. For a more
 * efficient search in a large hyperparameter space, consider implementing:
 * </p>
 * <ul>
 *   <li><b>Random Search:</b> Instead of testing all combinations, Random Search samples a fixed
 *       number of random combinations. It often finds better models in less time.</li>
 *   <li><b>Bayesian Optimization:</b> An even more advanced technique that uses the results from
 *       previous trials to inform which combination to try next, converging on the optimal
 *       parameters more quickly.</li>
 * </ul>
 *
 * @see MLP23
 * @see MLP
 * @see ExecutorService
 * @author Brandon Mejia
 * @version 2025-11-29
 */
public class HyperparameterTuner {

    // --- Dados de treino ---
    private final String[] inputPaths = {
            "src/data/borroso.csv",
            //"src/data/big.csv",
            "src/data/dataset.csv"
    };
    private final String[] outputPaths = {
            "src/data/labels.csv",
            //"src/data/labels.csv",
            "src/data/labels.csv"
    };

    // --- Hiperparámetros para a busca ---
    private final double[] learningRates = {0.01, 0.005, 0.001,0.02,0.03,0.1,0.002,0.0221,0.003,0.0005,0.0001,0.0002};
    private final double[] momentums = {0.6,0.7, 0.8, 0.9};
    private final int[][] topologies = {
            {400, 1, 1},
            {400, 2, 1},
            {400, 3, 1},
            {400, 4, 1}

    };
    private final IDifferentiableFunction[][] activationFunctions = {
            {new Sigmoid(), new Sigmoid()},
            {new TanH(), new TanH()},
            {new ReLU(), new Sigmoid()} // ReLU para camadas ocultas, Sigmoid para a saída
    };

    /**
     * A simple data class to store the results of a single training trial.
     */
    private static class TuningResult implements Comparable<TuningResult> {
        final String paramsDescription;
        final double validationError;

        TuningResult(String paramsDescription, double validationError) {
            this.paramsDescription = paramsDescription;
            this.validationError = validationError;
        }

        @Override
        public int compareTo(TuningResult other) {
            return Double.compare(this.validationError, other.validationError);
        }

        @Override
        public String toString() {
            return String.format("Parameters: [%s] -> Validation Error: %.6f", paramsDescription, validationError);
        }
    }

    /**
     * Executes a parallel grid search to find the best hyperparameter combination.
     * <p>
     * This method creates a list of all possible hyperparameter combinations and
     * uses a fixed-size thread pool to train and evaluate them in parallel.
     * Once all trials are complete, it prints a sorted summary of the results.
     * </p>
     */
    public void runGridSearch() {
        // Create all combinations of parameters to be tested.
        List<Callable<TuningResult>> tasks = new ArrayList<>();
        for (int[] topology : topologies) {
            for (IDifferentiableFunction[] functions : activationFunctions) {
                // Ensure the topology and activation function combination is valid.
                if (topology.length - 1 != functions.length) continue;

                for (double lr : learningRates) {
                    for (double momentum : momentums) {
                        // Create a task for the current combination.
                        tasks.add(() -> runTrial(topology, functions, lr, momentum));
                    }
                }
            }
        }

        // Use a fixed-size thread pool to run tasks in parallel.
        // Using the number of available processors is a good default.
        int numThreads = Runtime.getRuntime().availableProcessors();
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        // CompletionService decouples task submission from result retrieval, which is more efficient.
        CompletionService<TuningResult> completionService = new ExecutorCompletionService<>(executor);
        List<TuningResult> results = new ArrayList<>();

        System.out.printf("--- Starting Hyperparameter Search with %d threads ---\n", numThreads);
        System.out.printf("Total combinations to test: %d\n\n", tasks.size());

        // Submit all tasks for execution.
        for (Callable<TuningResult> task : tasks) {
            completionService.submit(task);
        }

        try {
            // Retrieve results as they become available.
            for (int i = 0; i < tasks.size(); i++) {
                try {
                    // Wait for the next completed task, with a generous timeout.
                    Future<TuningResult> future = completionService.poll(30, TimeUnit.MINUTES);
                    if (future == null) {
                        System.err.println("A training trial timed out and was cancelled.");
                        continue; // Move to the next result
                    }
                    results.add(future.get());
                } catch (CancellationException e) {
                    System.err.println("A training trial was cancelled, possibly due to a timeout.");
                } catch (ExecutionException e) {
                    System.err.println("A training trial failed: " + e.getCause().getMessage());
                    e.getCause().printStackTrace();
                }
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            System.err.println("Hyperparameter search was interrupted while waiting for a result.");
        } finally {
            executor.shutdown();
        }

        // Sort results by validation error (ascending).
        Collections.sort(results);

        // --- Print Summary ---
        System.out.println("\n\n--- === Search Summary === ---");
        results.forEach(System.out::println);

        if (!results.isEmpty()) {
            TuningResult bestResult = results.get(0);
            System.out.println("\n--- BEST COMBINATION FOUND ---");
            System.out.printf("Parameters: %s\n", bestResult.paramsDescription);
            System.out.printf("Best Validation Error (MSE): %.6f\n", bestResult.validationError);
        } else {
            System.out.println("\n--- No successful trials were completed. ---");
        }
        System.out.println("--- ======================== ---");
    }

    /**
     * Runs a single training and validation trial for a given set of hyperparameters.
     *
     * @return A {@link TuningResult} object containing the parameters and final validation error.
     */
    private TuningResult runTrial(int[] topology, IDifferentiableFunction[] functions, double lr, double momentum) {
        String currentParams = String.format(
                "Topology: %s, Functions: %s, LR: %.4f, Momentum: %.2f",
                Arrays.toString(topology),
                getFunctionNames(functions),
                lr,
                momentum
        );
        System.out.println("--- Testing combination: " + currentParams + " ---");

        MLP23 trainer = new MLP23(topology, functions, lr, momentum, 100000);
        double validationError = trainer.train(inputPaths, outputPaths);

        System.out.printf("--- Finished: [%s] -> Validation Error: %.6f ---\n", currentParams, validationError);
        return new TuningResult(currentParams, validationError);
    }

    /**
     * Helper to get the names of activation functions for logging.
     */
    private String getFunctionNames(IDifferentiableFunction[] functions) {
        List<String> names = new ArrayList<>();
        for (IDifferentiableFunction func : functions) {
            names.add(func.getClass().getSimpleName());
        }
        return String.join(", ", names);
    }

    /**
     * Entry point to run the optimizer.
     */
    public static void main(String[] args) {
        HyperparameterTuner tuner = new HyperparameterTuner();
        tuner.runGridSearch();
    }
}
