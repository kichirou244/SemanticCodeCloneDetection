import java.util.LinkedList;
import java.util.Queue;

public class BfsPathFinder {
    // Kiem tra duong di giua dinh goc va dinh dich tren do thi ma tran ke
    public static boolean hasPath(int[][] adjMatrix, int srcNode, int destNode) {
        int totalNodes = adjMatrix.length;
        boolean[] seen = new boolean[totalNodes];
        Queue<Integer> worklist = new LinkedList<>();

        seen[srcNode] = true;
        worklist.add(srcNode);

        while (!worklist.isEmpty()) {
            int node = worklist.poll();
            if (node == destNode) {
                return true;
            }
            for (int adj = 0; adj < totalNodes; adj++) {
                if (adjMatrix[node][adj] == 1 && !seen[adj]) {
                    seen[adj] = true;
                    worklist.add(adj);
                }
            }
        }
        return false;
    }
}
