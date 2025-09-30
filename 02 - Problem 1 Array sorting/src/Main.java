
import core.ArrayCfg;
import core.BestFirst;
import core.Board;
import core.GSolver;
import java.util.Iterator;
import java.util.Scanner;

/**
 * Main class for the Array Sorting problem, as specified in the assignment requirements.
 * This class is responsible for handling user input and printing the solution output.
 * It reads an initial and a goal array configuration, uses {@link GSolver} to find the
 * lowest-cost solution path, and prints the sequence of states and the total cost.
 *
 * @author Brandon Mejia
 * @version 2025-09-27
 */
public class Main
{
    public static void main (String [] args)
    {
        boolean gSolve = false;

        if(gSolve)
            GSolve();
        else
            bestFirst();
    }


    public static void GSolve()
    {
        Scanner sc = new Scanner(System.in);
        GSolver gs = new GSolver();
        Iterator<GSolver.State> it =
                gs.solve( new ArrayCfg(sc.nextLine()), new ArrayCfg(sc.nextLine()));
        if (it==null) System.out.println("no solution found");
        else {
            while(it.hasNext()) {
                GSolver.State i = it.next();
                System.out.println(i);
                if (!it.hasNext()) System.out.println((int)i.getK());
            }
        }
        sc.close();

    }


    public static void bestFirst()
    {
        Scanner sc = new Scanner(System.in);

        BestFirst s = new BestFirst();
        Iterator<BestFirst.State> it = s.solve(new Board(sc.next()),
                new Board(sc.next()));
        if (it==null) System.out.println("no solution found");
        else {
            while(it.hasNext()) {
                BestFirst.State i = it.next();
                System.out.println(i);

                if (!it.hasNext()) System.out.println((int)i.getG());
            }
        }
        sc.close();
    }
}
