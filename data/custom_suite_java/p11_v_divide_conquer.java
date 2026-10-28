public class MaxSubarrayDivideConquer {
    public static int maxSubArray(int[] nums) {
        return solve(nums, 0, nums.length - 1);
    }
    private static int solve(int[] nums, int l, int r) {
        if (l == r) return nums[l];
        int m = l + (r - l) / 2;
        int leftMax = solve(nums, l, m);
        int rightMax = solve(nums, m + 1, r);
        int crossMax = crossSum(nums, l, m, r);
        return Math.max(Math.max(leftMax, rightMax), crossMax);
    }
    private static int crossSum(int[] nums, int l, int m, int r) {
        int leftSum = Integer.MIN_VALUE, sum = 0;
        for (int i = m; i >= l; i--) { sum += nums[i]; leftSum = Math.max(leftSum, sum); }
        int rightSum = Integer.MIN_VALUE; sum = 0;
        for (int i = m + 1; i <= r; i++) { sum += nums[i]; rightSum = Math.max(rightSum, sum); }
        return leftSum + rightSum;
    }
}
