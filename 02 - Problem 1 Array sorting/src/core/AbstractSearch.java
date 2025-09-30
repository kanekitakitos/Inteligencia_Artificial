
package core;

import java.util.*;

public abstract class AbstractSearch {
    protected Queue<State> abertos;
    protected Map<Ilayout, State> fechados; // Using this map to track the best state found for a layout
    protected State actual;
    protected Ilayout objective;
    private static long sequenceCounter = 0;

    public static class State {
        private final Ilayout layout;
        private final State father;
        private final double g; // Cost from start to current state
        private final long sequenceId;

        public State(Ilayout l, State n) {
            layout = l;
            father = n;
            sequenceId = AbstractSearch.sequenceCounter++;

            if (father != null) {
                g = father.g + layout.getK();
            } else {
                g = 0.0;
            }
        }

        @Override
        public String toString() {
            return layout.toString();
        }

        public double getK() { return g; }
        public double getG() { return g; }
        public long getSequenceId() { return sequenceId; }
        public Ilayout getLayout() { return layout; }
        public State getFather() { return father; }

        @Override
        public int hashCode() { return layout.hashCode(); }

        @Override
        public boolean equals(Object o) {
            if (o == null || getClass() != o.getClass()) return false;
            State n = (State) o;
            return this.layout.equals(n.layout);
        }
    }

    protected final List<State> generateSuccessors(State n) {
        List<State> sucs = new ArrayList<>();
        List<Ilayout> children = n.layout.children();
        for (Ilayout e : children) {
            sucs.add(new State(e, n));
        }
        return sucs;
    }

    public final Iterator<State> solve(Ilayout s, Ilayout goal)
    {
        objective = goal;
        abertos = createFringe();
        fechados = new HashMap<>();
        sequenceCounter = 0;

        State initialState = new State(s, null);
        abertos.add(initialState);
        fechados.put(initialState.getLayout(), initialState);

        while (!abertos.isEmpty()) {
            actual = abertos.poll();


            State currentBest = fechados.get(actual.getLayout());
            if (actual.getG() > currentBest.getG()) {
                System.err.println("SKIPPING stale node=" + actual.getLayout() + " seq=" + actual.getSequenceId() + " totalG=" + actual.getG() + " (better path exists with G=" + currentBest.getG() + ")");
                continue;
            }

            System.err.println("--------------------------------------------------------------------------------------------------------------------");
            System.err.println("EXPANDING node=" + actual.getLayout() + " totalG=" + actual.getG() + " seq=" + actual.getSequenceId());

            if (actual.getLayout().isGoal(objective)) {
                System.err.println("GOAL FOUND!");
                LinkedList<State> path = new LinkedList<>();
                State current = actual;
                while (current != null) {
                    path.addFirst(current);
                    current = current.getFather();
                }
                return path.iterator();
            }

            List<State> sucs = generateSuccessors(actual);
            int childNum = 0;
            for (State successor : sucs) {
                State existing = fechados.get(successor.getLayout());


                if (existing == null || successor.getG() < existing.getG()) {
                    fechados.put(successor.getLayout(), successor);
                    abertos.add(successor);
                    System.err.println("  CHILD#" + childNum + " ADD child= " + successor.getLayout() + " step= " + successor.getLayout().getK() + " totalG= " + successor.getG() + " seq=" + successor.getSequenceId());
                } else {
                    System.err.println("  CHILD#" + childNum + " IGNORE = " + successor.getLayout() + " step= " + successor.getLayout().getK() + " totalG= " + successor.getG() + " seq=" + successor.getSequenceId() + " (worse than existing seq=" + existing.getSequenceId() + " existingG=" + existing.getG() + ")");
                }
                childNum++;
            }
        }
        return null;
    }

    protected abstract Queue<State> createFringe();
}