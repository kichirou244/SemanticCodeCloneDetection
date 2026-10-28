int search(int numbers[], int count, int key) {
    int leftIdx = 0, rightIdx = count - 1;
    while (leftIdx <= rightIdx) {
        int middle = leftIdx + (rightIdx - leftIdx) / 2;
        if (numbers[middle] == key) return middle;
        if (numbers[middle] < key) leftIdx = middle + 1;
        else rightIdx = middle - 1;
    }
    return -1;
}
