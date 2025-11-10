package apps;

import math.Matrix;
import neural.activation.IDifferentiableFunction;
import neural.MLP;
import neural.activation.Sigmoid;
import neural.activation.Step;
import java.util.Scanner;

/**
 * @author hdaniel@ualg.pt
 * @version 202511050300
 */
public class MLPNXOR {

    public static void main(String[] args) {

        double lr    = 0.4;  // learning rate.
        int    epochs = 5000; //define the number of epochs in the order of thousands
        int[] topology = {2, 2, 1};

        //Dataset
        Matrix trX = new Matrix(
                new double[][]{
                        {0, 0},
                        {0, 1},
                        {1, 0},
                        {1, 1}});

        Matrix trY = new Matrix(
                new double[][]{
                        {1},
                        {0},
                        {0},
                        {1}});


        //Get input and create evaluation Matrix
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        double[][] input = new double[n][2];
        for (int i = 0; i < n; i++) {
            input[i][0] = Double.parseDouble(sc.next().replace(',', '.'));
            input[i][1] = Double.parseDouble(sc.next().replace(',', '.'));
        }
        Matrix evX = new Matrix(input);

        //Train the MLP
        MLP mlp = new MLP(topology,
                new IDifferentiableFunction[]{
                        new Sigmoid(),
                        new Sigmoid(),},
                -1);
        mlp.train(trX, trY, lr, epochs);

        //Predict and output results
        Matrix pred = mlp.predict(evX);

        //convert probabilities to integer classes: 0 or 1
        pred = pred.apply(new Step().fnc());

        //print output
        for (int i = 0; i < pred.rows(); i++) {
            System.out.printf("%d\n", (int) pred.get(i, 0));
        }

        sc.close();

    }
}
