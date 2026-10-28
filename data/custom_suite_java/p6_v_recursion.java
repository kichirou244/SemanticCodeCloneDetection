public class RecursiveBinarySearch {
    public static int search(int[] arr, int target) {
        return bsearch(arr, target, 0, arr.length - 1);
    }
    private static int bsearch(int[] arr, int target, int low, int high) {
        if (low > high) return -1;
        int mid = low + (high - low) / 2;
        if (arr[mid] == target) return mid;
        if (arr[mid] < target) return bsearch(arr, target, mid + 1, high);
        return bsearch(arr, target, low, mid - 1);
    }
}
