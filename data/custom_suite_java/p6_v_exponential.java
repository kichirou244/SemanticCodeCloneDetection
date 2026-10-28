public class ExponentialSearch {
    public static int search(int[] arr, int target) {
        int n = arr.length;
        if (n == 0) return -1;
        if (arr[0] == target) return 0;
        int i = 1;
        while (i < n && arr[i] <= target) i = i * 2;
        int low = i / 2;
        int high = Math.min(i, n - 1);
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (arr[mid] == target) return mid;
            if (arr[mid] < target) low = mid + 1;
            else high = mid - 1;
        }
        return -1;
    }
}
