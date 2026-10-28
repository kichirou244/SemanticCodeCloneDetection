import java.util.LinkedList;
import java.util.Queue;

public class GraphBfs {
    public static boolean hasPath(int[][] graph, int start, int target) {
        int n = graph.length;
        boolean[] visited = new boolean[n];
        Queue<Integer> queue = new LinkedList<>();

        visited[start] = true;
        queue.add(start);

        while (!queue.isEmpty()) {
            int current = queue.poll();
            if (current == target) {
                return true;
            }
            for (int neighbor = 0; neighbor < n; neighbor++) {
                if (graph[current][neighbor] == 1 && !visited[neighbor]) {
                    visited[neighbor] = true;
                    queue.add(neighbor);
                }
            }
        }
        return false;
    }
}
