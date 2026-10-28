#include <vector>
#include <algorithm>
#include <climits>
using namespace std;
int crossSum(const vector<int>& nums, int l, int m, int r) {
    int leftSum = INT_MIN, sum = 0;
    for (int i = m; i >= l; i--) { sum += nums[i]; leftSum = max(leftSum, sum); }
    int rightSum = INT_MIN; sum = 0;
    for (int i = m + 1; i <= r; i++) { sum += nums[i]; rightSum = max(rightSum, sum); }
    return leftSum + rightSum;
}
int solve(const vector<int>& nums, int l, int r) {
    if (l == r) return nums[l];
    int m = l + (r - l) / 2;
    int leftMax = solve(nums, l, m);
    int rightMax = solve(nums, m + 1, r);
    int crossMax = crossSum(nums, l, m, r);
    return max({leftMax, rightMax, crossMax});
}
int maxSubArray(const vector<int>& nums) {
    return solve(nums, 0, (int)nums.size() - 1);
}
