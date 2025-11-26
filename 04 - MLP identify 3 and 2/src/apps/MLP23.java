package apps;

import math.Matrix;
import neural.activation.IDifferentiableFunction;
import neural.activation.Sigmoid;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Scanner;
import neural.MLP;



public class MLP23 {
    private double lr = 1;  // Taxa de aprendizado (0 < lr < 1)
    private int epochs = 10000; // Número de épocas de treinamento
    private int[] topology = {400, 16,16, 1}; // Topologia: 400 entradas, 5 neurônios na camada oculta, 1 na saída
    private MLP mlp;
    private int seek = 4;

    /**
     * Classe interna para agrupar uma amostra de entrada com seu rótulo de saída.
     */
    private static class DataPoint {
        final double[] input;
        final double[] output;

        DataPoint(double[] input, double[] output) { this.input = input; this.output = output; }
    }

    public MLP23() {
        this.mlp = new MLP(topology,
                new IDifferentiableFunction[]{
                        new Sigmoid(),
                        new Sigmoid(),new Sigmoid()}, -1);
    }


    public MLP23(int seed) {
        this.mlp = new MLP(topology,
                new IDifferentiableFunction[]{
                        new Sigmoid(),
                        new Sigmoid(),new Sigmoid()}, seek);
    }


    /**
     * Carrega os dados de um arquivo CSV.
     * @param filePath O caminho para o arquivo CSV.
     * @return Uma lista de arrays de double, onde cada array representa uma linha.
     */
    private List<double[]> loadData(String filePath) {
        List<double[]> data = new ArrayList<>();
        String line;
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            while ((line = br.readLine()) != null) {
                String[] stringValues = line.split(",");
                double[] doubleValues = new double[stringValues.length];
                for (int i = 0; i < stringValues.length; i++) {
                    doubleValues[i] = Double.parseDouble(stringValues[i]);
                }
                data.add(doubleValues);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return data;
    }

    /**
     * Treina a rede neural com os datasets fornecidos.
     * Os rótulos '2' são mapeados para 0 e '3' para 1.
     * @param inputPath Caminho para o arquivo de dados de entrada.
     * @param outputPath Caminho para o arquivo de rótulos de saída.
     */
    public void train(String inputPath, String outputPath) {
        // 1. Carrega os dados de entrada e saída
        List<double[]> inputDataList = loadData(inputPath);
        List<double[]> outputDataList = loadData(outputPath);

        // 2. Combina entradas e saídas em uma única lista e normaliza os rótulos
        List<DataPoint> combinedData = new ArrayList<>();
        for (int i = 0; i < inputDataList.size(); i++) {
            double[] currentOutput = outputDataList.get(i);
            // Normaliza a saída: 2.0 -> 0.0, 3.0 -> 1.0
            if (currentOutput[0] == 2.0) {
                currentOutput[0] = 0.0;
            } else if (currentOutput[0] == 3.0) {
                currentOutput[0] = 1.0;
            }
            combinedData.add(new DataPoint(inputDataList.get(i), currentOutput));
        }

        // 3. Embaralha a lista combinada para randomizar a ordem de treinamento
        // A linha abaixo foi comentada para permitir o teste com dados ordenados.
        // Desativar o embaralhamento pode impactar negativamente a capacidade de generalização do modelo.
         Collections.shuffle(combinedData, new Random(seek));
        Collections.shuffle(combinedData, new Random(seek+1));
        // 4. Separa os dados embaralhados de volta em arrays para criar as Matrizes
        double[][] shuffledInputs = new double[combinedData.size()][];
        double[][] shuffledOutputs = new double[combinedData.size()][];
        for (int i = 0; i < combinedData.size(); i++) {
            shuffledInputs[i] = combinedData.get(i).input;
            shuffledOutputs[i] = combinedData.get(i).output;
        }

        // (NOVO) Salva os dados que serão usados para o treinamento em um arquivo para verificação
        //saveTrainingDataForVerification(shuffledInputs, shuffledOutputs, "src/data/training_verification.txt");

        // Salva os pesos iniciais antes do treinamento
        saveWeightsAndBiases(mlp, "--- Pesos Iniciais (Antes do Treinamento) ---", "src/data/weights_log.txt", false);

        // 5. Chama o método de treinamento com os dados embaralhados
        System.out.println("Iniciando o treinamento da rede...");
        mlp.train(new Matrix(shuffledInputs), new Matrix(shuffledOutputs), this.lr, this.epochs);
        System.out.println("Treinamento concluído.");

        // Salva os pesos finais após o treinamento
        saveWeightsAndBiases(mlp, "\n--- Pesos Finais (Após o Treinamento) ---", "src/data/weights_log.txt", true);
    }

    /**
     * Salva os pesos e biases da MLP em um arquivo de texto para análise.
     *
     * @param mlp      A instância da MLP.
     * @param title    Um título para a seção (ex: "Pesos Iniciais").
     * @param filePath O caminho do arquivo para salvar os dados.
     * @param append   Se `true`, adiciona ao final do arquivo; se `false`, sobrescreve o arquivo.
     */
    private void saveWeightsAndBiases(MLP mlp, String title, String filePath, boolean append) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filePath, append))) {
            writer.println(title);
            writer.println("-".repeat(title.length()));

            Matrix[] weights = mlp.getWeights();
            Matrix[] biases = mlp.getBiases();

            for (int l = 0; l < weights.length; l++) { // Itera sobre cada camada (da entrada para a saída)
                writer.printf("\n[Camada %d -> Camada %d]\n", l, l + 1);
                Matrix layerWeights = weights[l]; // Pesos que conectam a camada l com a l+1
                Matrix layerBiases = biases[l];   // Biases dos neurônios na camada l+1

                for (int j = 0; j < layerWeights.cols(); j++) { // Itera sobre cada neurônio na camada l+1
                    writer.printf("  Neurônio %d:\n", j);
                    writer.printf("    - Bias: % .8f\n", layerBiases.get(0, j));
                    for (int i = 0; i < layerWeights.rows(); i++) { // Itera sobre as conexões do neurônio anterior (camada l)
                        writer.printf("    - Peso da conexão com Neurônio %d (camada %d): % .8f\n", i, l, layerWeights.get(i, j));
                    }
                }
            }
            System.out.println("Pesos salvos em: " + filePath);
        } catch (IOException e) {
            System.err.println("Erro ao salvar os pesos: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Salva os dados de entrada e saída, prontos para o treinamento, em um arquivo para verificação.
     * Isso permite inspecionar se o embaralhamento e a normalização dos rótulos ocorreram como esperado.
     *
     * @param inputs As matrizes de entrada (features).
     * @param outputs Os rótulos de saída correspondentes.
     * @param filePath O caminho do arquivo onde os dados serão salvos.
     */
    private void saveTrainingDataForVerification(double[][] inputs, double[][] outputs, String filePath) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filePath))) {
            writer.println("--- Dados de Treinamento para Verificação ---");
            writer.println("Total de Amostras: " + inputs.length);
            writer.println("---------------------------------------------");
            for (int i = 0; i < inputs.length; i++) {
                writer.printf("Amostra de Treino #%d: Rótulo Esperado = %.1f%n", i, outputs[i][0]);
            }
            System.out.println("Dados de treinamento para verificação salvos em: " + filePath);
        } catch (IOException e) {
            System.err.println("Erro ao salvar os dados de verificação do treinamento: " + e.getMessage());
            e.printStackTrace();
        }
    }

    public MLP getMLP() {
        return this.mlp;
    }

    public static void main(String[] args) {
        MLP23 model = new MLP23();

        // Caminhos para os seus arquivos de dados
        String inputPath = "src/data/dataset.csv";
        String outputPath = "src/data/labels.csv";

        // Inicia o processo de treinamento
        model.train(inputPath, outputPath);

        testar(model);


    }

    public static void testar(MLP23 model)
    {
        // --- Seção de Teste Interativo ---
        System.out.println("\n--- Modo de Teste Interativo ---");
        Scanner sc = new Scanner(System.in);
        int numTests = 0;

        while (true) {
            try {
                System.out.print("Digite o número de imagens que deseja testar: ");
                numTests = sc.nextInt();
                sc.nextLine(); // Consome a nova linha pendente
                break;
            } catch (Exception e) {
                System.err.println("Entrada inválida. Por favor, insira um número inteiro.");
                sc.nextLine(); // Limpa o buffer do scanner
            }
        }

        for (int i = 0; i < numTests; i++) {
            System.out.printf("\n--- Teste #%d ---\n", i + 1);
            System.out.println("Cole os 400 valores de píxeis separados por vírgula e pressione Enter:");
            String line = sc.nextLine();
            String[] stringValues = line.split(",");

            if (stringValues.length != 400) {
                System.err.printf("Erro: Foram inseridos %d valores, mas são esperados 400. Pulando este teste.\n", stringValues.length);
                continue;
            }

            double[][] testInput = new double[1][400];
            for (int j = 0; j < stringValues.length; j++) {
                testInput[0][j] = Double.parseDouble(stringValues[j].trim());
            }

            Matrix prediction = model.getMLP().predict(new Matrix(testInput));
            long predictedLabel = Math.round(prediction.get(0, 0));
            System.out.printf("Resultado da Predição: %d (0 para '2', 1 para '3')\n", predictedLabel);
        }
        sc.close();

    }
}
