#include <vector>
#include <algorithm>
using namespace std;
int maxSubArray(const vector<int>& numbers) {
    int bestSum = numbers[0], runningSum = numbers[0];
    for (size_t idx = 1; idx < numbers.size(); idx++) {
        runningSum = max(numbers[idx], runningSum + numbers[idx]);
        bestSum = max(bestSum, runningSum);
    }
    return bestSum;
}
