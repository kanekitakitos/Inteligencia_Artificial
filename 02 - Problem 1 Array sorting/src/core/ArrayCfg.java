package core;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Immutable array configuration implementing Ilayout.
 */
public final class ArrayCfg implements Ilayout {

    private final int[] data;
    private final int cost; // cost of the swap that produced this state (0 for initial)

    // Constructor used for initial/goal states (from a string).
    public ArrayCfg(String s) {
        if (s == null) throw new IllegalArgumentException("Input string cannot be null");
        String trimmed = s.trim();
        if (trimmed.isEmpty()) {
            this.data = new int[0];
        } else {
            // split on any whitespace sequence (handles multiple spaces, tabs, etc.)
            String[] parts = trimmed.split("\\s+");
            int[] parsed = Arrays.stream(parts).mapToInt(Integer::parseInt).toArray();
            this.data = Arrays.copyOf(parsed, parsed.length); // defensive copy
        }
        this.cost = 0;
    }

    // Private constructor for children
    private ArrayCfg(int[] data, int cost) {
        this.data = data; // already a copy by caller
        this.cost = cost;
    }

    @Override
    public List<Ilayout> children() {
        // Get the size of the array.
        int n = data.length;

        // 1. Handle Edge Case: If the array has 0 or 1 elements, no swaps are possible.
        //    Return an empty list immediately.
        if (n < 2) return Collections.emptyList();

        // 2. Pre-allocate Memory: Create a list to hold the children.
        //    The capacity is set to n * (n - 1) / 2, which is the exact number of
        //    unique pairs you can form from 'n' items. This is a performance
        //    optimization to prevent the list from having to resize itself as items are added.
        List<Ilayout> children = new ArrayList<>(n * (n - 1) / 2);

        // 3. Generate All Unique Swaps: These nested loops iterate through all
        //    unique pairs of indices (i, j) in the array.
        for (int i = 0; i < n - 1; i++)
        {
            // The inner loop starts from the end and goes down to just after 'i'.
            // This ensures that each pair of indices (like (0, 1) and (1, 0)) is
            // considered only once.
            for (int j = i + 1; j < n; j++)
            {

                // 4. Create a Child State: For each swap, create a new state.
                //    It's crucial to copy the parent's array so that the child's
                //    modifications don't affect the parent or other children.
                int[] childData = Arrays.copyOf(data, n);

                // 5. Perform the Swap: Swap the elements at indices i and j
                //    in the new child's array.
                int tmp = childData[i];
                childData[i] = childData[j];
                childData[j] = tmp;

                // 6. Calculate the Cost: The cost of the move (the swap) is calculated
                //    based on the values from the *parent's* array before the swap.
                int swapCost = calculateCost(data[i], data[j]);

                // 7. Add the New Child: Create a new ArrayCfg object representing this
                //    new state and add it to the list of children.
                children.add(new ArrayCfg(childData, swapCost));
            }
        }

        // 8. Return an Immutable List: Return a read-only version of the list.
        //    This is good practice as it prevents the list of children from being
        //    modified accidentally by other parts of the program.
        return Collections.unmodifiableList(children);
    }

    /**
     * Compute swap cost:
     * - even & even -> 2
     * - odd & odd -> 20
     * - mixed -> 11
     */
    private static int calculateCost(int a, int b) {
        // Using bitwise to check parity; works for negative numbers too.
        boolean aEven = (a & 1) == 0;
        boolean bEven = (b & 1) == 0;

        if (aEven && bEven) return 2;
        if (!aEven && !bEven) return 20;
        return 11;
    }

    @Override
    public boolean isGoal(Ilayout l) {
        if (!(l instanceof ArrayCfg)) return false;
        return Arrays.equals(this.data, ((ArrayCfg) l).data);
    }

    @Override
    public double getK() {
        return  this.cost;
    }

    @Override
    public String toString() {
        return Arrays.stream(data)
                .mapToObj(String::valueOf)
                .collect(Collectors.joining(" "));
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof ArrayCfg)) return false;
        ArrayCfg other = (ArrayCfg) o;
        return Arrays.equals(this.data, other.data);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(data);
    }

    // Optional: expose a defensive copy if needed elsewhere
    public int[] asArrayCopy() {
        return Arrays.copyOf(data, data.length);
    }
}
