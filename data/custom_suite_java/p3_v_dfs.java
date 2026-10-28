public class GraphDfs {
    public static boolean hasPath(int[][] graph, int start, int target) {
        boolean[] visited = new boolean[graph.length];
        return dfs(graph, start, target, visited);
    }

    private static boolean dfs(int[][] g, int curr, int target, boolean[] visited) {
        if (curr == target) {
            return true;
        }
        visited[curr] = true;
        for (int next = 0; next < g.length; next++) {
            if (g[curr][next] == 1 && !visited[next]) {
                if (dfs(g, next, target, visited)) {
                    return true;
                }
            }
        }
        return false;
    }
}
