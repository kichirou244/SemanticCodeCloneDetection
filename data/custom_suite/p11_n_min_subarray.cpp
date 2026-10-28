#include <vector>
#include <algorithm>
using namespace std;
int minSubArray(const vector<int>& nums) {
    int minSoFar = nums[0], currMin = nums[0];
    for (size_t i = 1; i < nums.size(); i++) {
        currMin = min(nums[i], currMin + nums[i]);
        minSoFar = min(minSoFar, currMin);
    }
    return minSoFar;
}
