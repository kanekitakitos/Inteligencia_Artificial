package core;
import java.util.*;

/**
 * Implements a Uniform-Cost Search (UCS) algorithm for the array sorting problem.
 * This class extends {@link AbstractSearch} and provides the specific "fringe"
 * implementation required for UCS, which is a priority queue ordered by accumulated cost.
 *
 * @see Ilayout
 * @see AbstractSearch
 *
 * @author Brandon Mejia
 * @version 2025-09-27
 */
public class GSolver extends AbstractSearch
{

    @Override
    protected Queue<State> createFringe() 
    {
        return new PriorityQueue<>(
                (s1, s2) -> {
                    int costCompare = Double.compare(s1.getG(), s2.getG());
                    if (costCompare != 0) return costCompare;
                    // Break ties using sequence ID for consistent ordering
                    return Long.compare(s1.getSequenceId(), s2.getSequenceId());
                }
        );
    }
}
