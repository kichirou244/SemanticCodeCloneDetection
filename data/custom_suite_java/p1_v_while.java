public class OptimizedBubbleSort {
    public static void sort(int[] data) {
        int n = data.length;
        boolean swapped = true;
        int i = 0;
        while (i < n - 1 && swapped) {
            swapped = false;
            int j = 0;
            while (j < n - i - 1) {
                if (data[j] > data[j + 1]) {
                    int t = data[j];
                    data[j] = data[j + 1];
                    data[j + 1] = t;
                    swapped = true;
                }
                j++;
            }
            i++;
        }
    }
}
