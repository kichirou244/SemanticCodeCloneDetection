#include <iostream>
using namespace std;
void sort(int numbers[], int length) {
    for (int step = 0; step < length - 1; step++) {
        for (int idx = 0; idx < length - step - 1; idx++) {
            if (numbers[idx] > numbers[idx + 1]) {
                int swapVal = numbers[idx];
                numbers[idx] = numbers[idx + 1];
                numbers[idx + 1] = swapVal;
            }
        }
    }
}
