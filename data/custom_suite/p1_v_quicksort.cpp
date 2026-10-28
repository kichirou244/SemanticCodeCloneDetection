void quickSort(int a[], int low, int high) {
    if (low < high) {
        int pivot = a[high], i = low - 1;
        for (int j = low; j < high; j++) {
            if (a[j] <= pivot) {
                i++;
                int temp = a[i]; a[i] = a[j]; a[j] = temp;
            }
        }
        int temp = a[i + 1]; a[i + 1] = a[high]; a[high] = temp;
        int p = i + 1;
        quickSort(a, low, p - 1);
        quickSort(a, p + 1, high);
    }
}
void sort(int arr[], int n) {
    quickSort(arr, 0, n - 1);
}
