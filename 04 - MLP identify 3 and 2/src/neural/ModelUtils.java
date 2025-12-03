package neural;

import java.io.*;

/**
 * A utility class for handling model serialization and deserialization.
 * <p>
 * This class provides static methods to save and load {@link MLP} models to and from the file system.
 * It centralizes the file I/O logic for model persistence, ensuring a consistent approach across the application.
 * </p>
 * <h3>Example Usage</h3>
 * <pre>{@code
 * // To save a model:
 * MLP myModel = new MLP(...);
 * ModelUtils.saveModel(myModel, "src/models/my_model.ser");
 *
 * // To load a model:
 * MLP loadedModel = ModelUtils.loadModel("src/models/my_model.ser");
 * if (loadedModel != null) {
 *     // Use the model for predictions
 * }
 * }</pre>
 *
 * @see MLP
 * @author Brandon Mejia
 * @version 2025-12-03
 */
public class ModelUtils {

    /**
     * Loads a pre-trained MLP model from a file.
     *
     * @param filePath The path to the serialized model file.
     * @return A new {@link MLP} instance with the loaded state, or {@code null} if loading fails.
     */
    public static MLP loadModel(String filePath) {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath))) {
            return (MLP) ois.readObject();
        } catch (IOException | ClassNotFoundException e) {
            System.err.println("ERRO: Falha ao carregar o modelo de " + filePath);
            e.printStackTrace();
            return null;
        }
    }

    /**
     * Saves the current state of an MLP model to a file.
     *
     * @param model    The MLP model to be saved.
     * @param filePath The path to the file where the model will be saved.
     */
    public static void saveModel(MLP model, String filePath) {
        new File(filePath).getParentFile().mkdirs();
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath))) {
            oos.writeObject(model);
        } catch (IOException e) {
            System.err.println("ERRO: Falha ao guardar o modelo em " + filePath);
            e.printStackTrace();
        }
    }
}