public class ArrayMerger {
    // Tron 2 mang da sap xep thanh mang moi
    public static int[] merge(int[] leftArr, int[] rightArr) {
        int[] merged = new int[leftArr.length + rightArr.length];
        int p1 = 0, p2 = 0, idx = 0;
        while (p1 < leftArr.length && p2 < rightArr.length) {
            if (leftArr[p1] <= rightArr[p2]) merged[idx++] = leftArr[p1++];
            else merged[idx++] = rightArr[p2++];
        }
        while (p1 < leftArr.length) merged[idx++] = leftArr[p1++];
        while (p2 < rightArr.length) merged[idx++] = rightArr[p2++];
        return merged;
    }
}
