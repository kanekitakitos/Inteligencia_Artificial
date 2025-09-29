package core;
import java.util.*;

/**
 * Implements the Uniform-Cost Search (UCS) algorithm for the array sorting problem.
 * This class extends {@link AbstractSearch} and provides the specific "fringe"
 * implementation required for UCS, which is a priority queue ordered by accumulated cost
 * and a tie-breaking sequence number.
 * This is the main solver class used by the `Main` application.
 *
 * @see Ilayout
 * @see AbstractSearch
 *
 * @author Brandon Mejia
 * @version 2025-09-27
 */
public class GSolver extends AbstractSearch
{
    /**
     * Creates the fringe (the open set) for the Uniform-Cost Search algorithm.
     * @return A {@link PriorityQueue} that orders states first by their accumulated cost (`g`),
     * and then by their sequence number to ensure stability (FIFO for same-cost states).
     */
    @Override
    protected Queue<State> createFringe() {
        // Use a compound comparator: first by cost (g), then by sequence number for stability.
        Comparator<State> comparator = Comparator.comparingDouble(State::getG)
                                                 .thenComparingLong(State::getSequenceNumber);
        return new PriorityQueue<>(comparator);
    }
}
