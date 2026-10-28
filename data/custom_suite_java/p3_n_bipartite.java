import java.util.LinkedList;
import java.util.Queue;

public class GraphBipartiteCheck {
    public static boolean isBipartite(int[][] graph, int start, int ignored) {
        int n = graph.length;
        int[] colors = new int[n];
        Queue<Integer> queue = new LinkedList<>();

        colors[start] = 1;
        queue.add(start);

        while (!queue.isEmpty()) {
            int node = queue.poll();
            for (int neighbor = 0; neighbor < n; neighbor++) {
                if (graph[node][neighbor] == 1) {
                    if (colors[neighbor] == 0) {
                        colors[neighbor] = -colors[node];
                        queue.add(neighbor);
                    } else if (colors[neighbor] == colors[node]) {
                        return false;
                    }
                }
            }
        }
        return true;
    }
}
