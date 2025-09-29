package core;

import java.util.*;

/**
 * <h3>Implementing Search Strategies</h3>
 * <p>
 * This class uses the Template Method pattern. To implement a specific search algorithm,
 * you must extend this class and provide an implementation for the {@link #createFringe()}
 * method. The type of {@link Queue} you return determines the search strategy.
 * </p>
 *
 * <h4>Uniform-Cost Search (UCS)</h4>
 * <p>
 * Expands the node with the lowest path cost (g). Guarantees the least-cost path.
 * </p>
 * <ul>
 *   <li><b>Data Structure:</b> {@link java.util.PriorityQueue}</li>
 *   <li><b>Comparator:</b> Order by `State::getG`. For stable sorting (FIFO for same-cost nodes),
 *       use a secondary comparison on `State::getSequenceNumber`.
 *       <pre>{@code
 *       Comparator<State> comparator = Comparator.comparingDouble(State::getG)
 *                                                .thenComparingLong(State::getSequenceNumber);
 *       return new PriorityQueue<>(comparator);
 *       }</pre>
 *   </li>
 * </ul>
 *
 * <h4>Breadth-First Search (BFS)</h4>
 * <p>
 * Expands nodes level by level. Guarantees the shortest path in terms of number of steps,
 * assuming all step costs are equal.
 * </p>
 * <ul>
 *   <li><b>Data Structure:</b> A simple FIFO queue, like {@link java.util.LinkedList}.</li>
 *   <li><b>Implementation:</b>
 *       <pre>{@code return new LinkedList<>();}</pre>
 *   </li>
 * </ul>
 *
  * <h4>Depth-First Search (DFS)</h4>
 * <p>
 * Expands the deepest node first. It's not optimal and may not find a solution if the
 * search space has infinite branches.
 * </p>
 * <ul>
 *   <li><b>Data Structure:</b> A LIFO stack, which can be implemented with {@link java.util.ArrayDeque}.</li>
 *   <li><b>Implementation:</b> Since the `solve` method uses `queue.add(e)`, which adds to the end,
 *       we must override it to add to the front to simulate a stack's `push` operation.
 *       <pre>{@code
 *       return new ArrayDeque<>() {
 *           @Override
 *           public boolean add(State state) {
 *               this.addFirst(state);
 *               return true;
 *           }
 *       };
 *       }</pre>
 *   </li>
 * </ul>
 * 
 * @preConditions - Subclasses must implement the `createFringe()` method to define a search
 *   strategy. - The `solve` method must be called with valid, non-null initial and
 *   goal layouts.
 * @postConditions - The `solve` method returns an iterator for the solution path if one is
 *   found, or null otherwise. - The state of the search (open/closed lists) is
 *   managed internally.
 *
 * 
 * 
 *
 * @see Ilayout
 * @see State
 *
 * @author Brandon Mejia
 * @version 2025-09-27
 */
public abstract class AbstractSearch
{
    protected Queue<State> abertos;      // The fringe (open list) of states yet to be explored. The search strategy is defined by how this queue is ordered.
    protected Map<Ilayout, State> fechados; // The set of states already explored (closed list), used to prevent cycles and redundant work.
    protected State actual;             // The current state being processed, taken from the fringe.
    protected Ilayout objective;        // The goal state the algorithm is trying to reach.

    /**
     * Represents a node in the search tree, wrapping a layout and maintaining search-related information.
     * Each state holds a reference to its parent, allowing for path reconstruction, and stores the
     * accumulated cost (`g`) from the initial state to this node.
     */
    public static class State {
        private static long sequenceCounter = 0; // A global counter to assign a unique, incremental ID to each state upon creation.
        private final Ilayout layout;            // The problem-specific configuration (e.g., the array or board).
        private final State father;              // The parent state from which this state was generated, used to reconstruct the path.
        private final double g;                  // Accumulated cost of the path from the start to this state.
        private final long sequenceNumber;       // A unique ID for tie-breaking, ensuring FIFO order for states with the same cost.

        /**
         * Constructs a search node.
         * The accumulated cost `g` is calculated by adding the parent's cost
         * to the cost of the step leading to the current layout.
         * @param l The layout for this state.
         * @param n The parent state in the search tree.
         */
        public State(Ilayout l, State n) {
            layout = l;
            father = n;
            if (father != null)
                g = father.g + layout.getK(); // Use getK() from Ilayout for step cost
            else g = 0.0;
            this.sequenceNumber = sequenceCounter++;
        }

        @Override
        public String toString() {
            return layout.toString();
        }

        /**
         * Gets the total accumulated cost from the initial state to this state.
         * This is used by the Main class to print the final solution cost.
         * @return The total path cost (g).
         */
        public double getK() {
            return g;
        }

        /**
         * Gets the total accumulated cost from the initial state to this state.
         * This is used internally by search strategies (e.g., in a PriorityQueue) for ordering.
         * @return The total path cost (g).
         */
        public double getG() {
            return g;
        }

        /** @return The layout of this state. */
        public Ilayout getLayout() { return layout; }
        /** @return The parent state in the search path. */
        public State getFather() { return father; }
        /** @return The sequence number for tie-breaking. */
        public long getSequenceNumber() { return sequenceNumber; }

        @Override
        public int hashCode() { return layout.hashCode(); }

        @Override
        public boolean equals(Object o)
        {
            if (o == null || this.getClass() != o.getClass()) return false;
            State n = (State) o;
            return this.layout.equals(n.layout);
        }
    }

    /**
     * Generates the successor states for a given search node.
     * This method is common to most graph search algorithms.
     * @param n The parent node.
     * @return A list of `State` objects representing the children.
     */
    protected final List<State> generateSuccessors(State n)
    {
        List<State> sucs = new ArrayList<>();
        List<Ilayout> children = n.layout.children();

        for (Ilayout e : children)
            sucs.add(new State(e, n));
        return sucs;
    }

    /**
     * The "Template Method". It defines the invariant skeleton of the search algorithm.
     * It uses the `createFringe` hook method to allow subclasses to define
     * the search strategy.
     * @param s The initial layout of the problem.
     * @param goal The goal layout of the problem.
     * @return An iterator over the states of the solution path, or null if no solution is found.
     */
    public final Iterator<State> solve(Ilayout s, Ilayout goal)
    {
        objective = goal;
        abertos = createFringe(); // Hook method: subclasses define the fringe type/ordering
        // Initialize with a reasonable capacity to avoid early resizing
        fechados = new HashMap<>(1024);
        abertos.add(new State(s, null));

        while (!abertos.isEmpty()) {
            actual = abertos.poll();

            if (fechados.containsKey(actual.getLayout())) { // Skip if already processed via a cheaper path
                continue;
            }

            if (actual.getLayout().isGoal(objective)) {
                LinkedList<State> path = new LinkedList<>();
                State current = actual;
                while (current != null) {
                    path.addFirst(current);
                    current = current.getFather();
                }
                return path.iterator();
            }

            fechados.put(actual.getLayout(), actual);
            List<State> sucs = generateSuccessors(actual);

            for (State successor : sucs)
                if (!fechados.containsKey(successor.getLayout())) { // Only add if not already in closed list
                    abertos.add(successor);
                }

        }
        return null; // No solution found
    }

  /**
   * The "hook" method that subclasses must implement. This method defines the search
   * strategy by providing a specific type of queue (fringe).
   * @return A Queue implementation that dictates the search order.
   */
  protected abstract Queue<State> createFringe();
}