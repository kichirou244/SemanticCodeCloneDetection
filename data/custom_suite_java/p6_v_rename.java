public class BSearchAlgorithm {
    // Tim kiem nhi phan tren day da sap xep
    public static int search(int[] numbers, int key) {
        int leftIdx = 0;
        int rightIdx = numbers.length - 1;
        while (leftIdx <= rightIdx) {
            int middle = leftIdx + (rightIdx - leftIdx) / 2;
            if (numbers[middle] == key) return middle;
            if (numbers[middle] < key) leftIdx = middle + 1;
            else rightIdx = middle - 1;
        }
        return -1;
    }
}
