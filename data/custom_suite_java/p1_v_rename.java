public class ArraySorter {
    // Sap xep noi bot mang so nguyen theo thu tu tang dan
    public static void sort(int[] numbers) {
        int length = numbers.length;
        for (int step = 0; step < length - 1; step++) {
            for (int idx = 0; idx < length - step - 1; idx++) {
                if (numbers[idx] > numbers[idx + 1]) {
                    int swapValue = numbers[idx];
                    numbers[idx] = numbers[idx + 1];
                    numbers[idx + 1] = swapValue;
                }
            }
        }
    }
}
