package apps;

/**
 * A utility class to prepare and consolidate datasets for the MLP.
 */
public class DataPreparation {

    public static void main(String[] args) {
        System.out.println("--- A iniciar a preparação dos dados ---");

        // 1. Definir os ficheiros de entrada e saída que quer juntar
        String[] inputFilesToConcatenate = {
                "src/data/dataset.csv",
                //"src/data/dataset_novos.csv",
                //"src/data/dataset_apenas_novos.csv",
                //"src/data/dataset_apenas_novos2.csv"
        };

        String[] labelFilesToConcatenate = {
                "src/data/labels.csv",
                //"src/data/labels.csv", // Repetido para corresponder aos inputs
                //"src/data/labels.csv", // Repetido para corresponder aos inputs
                //"src/data/labels.csv"  // Repetido para corresponder aos inputs
        };

        // 2. Chamar o método para concatenar e guardar os ficheiros
        //DataHandler.concatenateAndSaveCsv(inputFilesToConcatenate, "src/data/treino_inputs_concatenados.csv");
        //DataHandler.concatenateAndSaveCsv(labelFilesToConcatenate, "src/data/treino_labels_concatenados.csv");

        System.out.println("\n--- Preparação de dados concluída ---");
    }
}