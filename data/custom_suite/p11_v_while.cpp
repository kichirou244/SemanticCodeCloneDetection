#include <vector>
using namespace std;
int maxSubArray(const vector<int>& nums) {
    int maxSoFar = nums[0], currMax = nums[0];
    size_t i = 1;
    while (i < nums.size()) {
        if (currMax < 0) currMax = nums[i];
        else currMax += nums[i];
        if (currMax > maxSoFar) maxSoFar = currMax;
        i++;
    }
    return maxSoFar;
}
