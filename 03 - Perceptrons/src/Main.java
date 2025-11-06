

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

public class Main {
    public static void main(String[] args) {
        // Define the activation function (Step Function)
        Function<Double, Integer> stepFunction = sum -> (sum >= 0) ? 1 : 0;
        
        // Define the dataset for the logic y = x2, as per "Tarefa 1"
        double[][] inputs = { {0, 0},
                              {0, 1},
                              {1, 0},
                              {1, 1}};

        int[] expected = {0, 1, 0, 1};

        // --- Tarefa 4: Configurar e Testar a Rede Multicamada (MLP) ---
        System.out.println("\n--- Teste da Rede Multicamada (MLP) para y = (¬x1 AND x2) OR (x1 AND x2) ---");
        
        // 1. Definir a Camada Oculta (Neurónio A e Neurónio B)
        // Neurónio A = ¬x1 AND x2. Pesos: w1=-2, w2=2, bias=-1
        Neuron neuronA = new Neuron(new double[]{-2.0, 2.0}, -1.0, stepFunction);
        
        // Neurónio B = x1 AND x2. Pesos: w1=1, w2=1, bias=-2
        Neuron neuronB = new Neuron(new double[]{1.0, 1.0}, -2.0, stepFunction);
        
        // 2. Definir a Camada de Saída (Neurónio C)
        // Neurónio C = A OR B. Pesos: wA=1, wB=1, bias=-1
        // As entradas para este neurónio são as saídas da camada oculta (de A e B)
        Neuron neuronC = new Neuron(new double[]{1.0, 1.0}, -1.0, stepFunction);

        // 3. Instanciar e construir a Rede Neuronal de forma fluida
        NeuralNetwork mlp = new NeuralNetwork();
        mlp.addLayer(neuronA, neuronB); // Adiciona a camada oculta
        mlp.addLayer(neuronC);         // Adiciona a camada de saída
        
        // 5. Testar a rede
        printResultsHeader("MLP");
        for (int i = 0; i < inputs.length; i++)
        {
            int result = mlp.predict(inputs[i]);
            printResultRow(inputs[i], expected[i], result);
        }

        // 6. Visualizar a arquitetura da rede no final da execução
        mlp.visualize();
    }

    private static void printResultsHeader(String modelName) {
        System.out.println("x1 | x2 | Expected | " + modelName + " Result");
        System.out.println("---|----|----------|---------------");
    }

    private static void printResultRow(double[] input, int expected, int result) {
        System.out.printf("%2.0f | %2.0f | %-8d | %d%n", input[0], input[1], expected, result);
    }
}
