import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashSet;
import java.util.Set;

public class GraphBfsCollections {
    public static boolean hasPath(int[][] matrix, int startNode, int goalNode) {
        int size = matrix.length;
        Set<Integer> visitedSet = new HashSet<>();
        Deque<Integer> deque = new ArrayDeque<>();

        visitedSet.add(startNode);
        deque.offer(startNode);

        while (!deque.isEmpty()) {
            int u = deque.poll();
            if (u == goalNode) {
                return true;
            }
            for (int v = 0; v < size; v++) {
                if (matrix[u][v] == 1 && !visitedSet.contains(v)) {
                    visitedSet.add(v);
                    deque.offer(v);
                }
            }
        }
        return false;
    }
}
