void sort(int arr[], int n) {
    int i = 0;
    bool swapped = true;
    while (i < n - 1 && swapped) {
        swapped = false;
        int j = 0;
        while (j < n - i - 1) {
            if (arr[j] > arr[j + 1]) {
                int t = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = t;
                swapped = true;
            }
            j++;
        }
        i++;
    }
}
