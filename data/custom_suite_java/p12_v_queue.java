import java.util.PriorityQueue;

public class MergeWithHeap {
    public static int[] merge(int[] a, int[] b) {
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        for (int x : a) pq.add(x);
        for (int y : b) pq.add(y);
        int[] res = new int[a.length + b.length];
        int idx = 0;
        while (!pq.isEmpty()) res[idx++] = pq.poll();
        return res;
    }
}
