package apps;

import neural.MLP;

public class MLP23
{
    double lr    = 0.1;  // learning rate 0 > lr > 1
    int    epochs = 10000; //define the number of epochs in the order of thousands
    int[] topology = {400, 2, 1};
    MLP mlp = null;





    public MLP getMLP()
    {
        return this.mlp.clone();
    }

}
