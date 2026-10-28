public class MergeSortedFor {
    public static int[] merge(int[] a, int[] b) {
        int n = a.length, m = b.length;
        int[] res = new int[n + m];
        int i = 0, j = 0;
        for (int k = 0; k < n + m; k++) {
            if (i >= n) res[k] = b[j++];
            else if (j >= m) res[k] = a[i++];
            else if (a[i] <= b[j]) res[k] = a[i++];
            else res[k] = b[j++];
        }
        return res;
    }
}
