import java.util.LinkedList;
import java.util.Queue;

public class GraphBfsShortestPath {
    public static boolean hasPath(int[][] graph, int start, int target) {
        if (start == target) return true;
        int n = graph.length;
        int[] dist = new int[n];
        for (int i = 0; i < n; i++) dist[i] = -1;

        Queue<Integer> q = new LinkedList<>();
        dist[start] = 0;
        q.add(start);

        while (!q.isEmpty()) {
            int curr = q.poll();
            for (int next = 0; next < n; next++) {
                if (graph[curr][next] == 1 && dist[next] == -1) {
                    dist[next] = dist[curr] + 1;
                    if (next == target) {
                        return true;
                    }
                    q.add(next);
                }
            }
        }
        return dist[target] != -1;
    }
}
