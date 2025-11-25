package test;

import math.Matrix;
import apps.MLP23;
import neural.MLP;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

@DisplayName("MLP Test for identifying digits 2 and 3")
class TestMLP23 {

    private MLP mlp;
    private Matrix testX;
    private Matrix testY;


    /**
     * Loads data from the CSV file.
     * The first 100 rows are images of the digit '2' (label 0).
     * The next rows are images of the digit '3' (label 1).
     */
    private void loadData()
    {
        String testFile = "src/data/dataset.csv";
        List<double[]> featureList = new ArrayList<>(); // Changed from MLP to mlp
        List<double[]> labelList = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(testFile))) {
            String line;
            int lineCount = 0;
            while ((line = br.readLine()) != null) {
                String[] stringValues = line.split(",");
                double[] features = new double[stringValues.length];
                for (int i = 0; i < stringValues.length; i++) {
                    features[i] = Double.parseDouble(stringValues[i]);
                }
                featureList.add(features);

                // The first 100 rows are '2' (label 0), the rest are '3' (label 1)
                labelList.add(new double[]{(lineCount < 100) ? 0 : 1});
                lineCount++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        testX = new Matrix(featureList.toArray(new double[0][]));
        testY = new Matrix(labelList.toArray(new double[0][]));
    }

    @BeforeEach
    void setUp() {
        // Load the test data
        loadData();

        // Get a pre-trained clone of the MLP model
        MLP23 modelFactory = new MLP23();
        this.mlp = modelFactory.getMLP();
    }

    @Test
    @DisplayName("Should achieve high accuracy on the test set")
    void testModelAccuracy() {
        int correctPredictions = 0;
        for (int i = 0; i < testY.rows(); i++) {
            // Create a 1xN matrix for a single prediction
            double[] singleTest = new double[testX.cols()];
            for (int j = 0; j < testX.cols(); j++) {
                singleTest[j] = testX.get(i, j);
            }
            Matrix input = new Matrix(new double[][]{singleTest});

            Matrix prediction = mlp.predict(input);
            long predictedLabel = Math.round(prediction.get(0, 0));
            double actualValue = testY.get(i, 0);

            if (predictedLabel == actualValue) {
                correctPredictions++;
            }
        }

        double accuracy = (double) correctPredictions / testY.rows();
        System.out.printf("Model Accuracy: %.2f%%\n", accuracy * 100);

        // Assert that the accuracy is above a certain threshold, for example, 95%
        assertTrue(accuracy > 0.95, "Model accuracy should be greater than 95%");
    }

    @Test
    @DisplayName("Should correctly predict a '2'")
    void testSinglePredictionForDigitTwo() {
        // Using the first image, which is a '2' (label 0)
        Matrix inputForTwo = getRowAsMatrix(testX, 0);
        Matrix prediction = mlp.predict(inputForTwo);
        long predictedLabel = Math.round(prediction.get(0, 0));

        assertEquals(0, predictedLabel, "The MLP should predict '0' for an image of a '2'");
    }

    @Test
    @DisplayName("Should correctly predict a '3'")
    void testSinglePredictionForDigitThree() {
        // Using the 101st image, which is a '3' (label 1)
        Matrix inputForThree = getRowAsMatrix(testX, 100);
        Matrix prediction = mlp.predict(inputForThree);
        long predictedLabel = Math.round(prediction.get(0, 0));

        assertEquals(1, predictedLabel, "The MLP should predict '1' for an image of a '3'");
    }

    /**
     * Helper method to extract a single row from a matrix and return it as a new 1xN Matrix.
     * @param originalMatrix The matrix to extract the row from.
     * @param rowIndex The index of the row to extract.
     * @return A new 1xN matrix containing the data from the specified row.
     */
    private Matrix getRowAsMatrix(Matrix originalMatrix, int rowIndex) {
        double[] row = new double[originalMatrix.cols()];
        for (int j = 0; j < originalMatrix.cols(); j++) {
            row[j] = originalMatrix.get(rowIndex, j);
        }
        return new Matrix(new double[][]{row});
    }
}